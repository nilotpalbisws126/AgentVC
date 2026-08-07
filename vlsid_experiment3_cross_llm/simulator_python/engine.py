"""
Discrete-Event Simulation Engine for Vehicular Cloud Scheduling.
Orchestrates task arrivals, vehicle arrivals/departures, subtask execution,
and policy hooks for all comparative algorithms.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional
from simulator_python.trace_loader import CarTrace, TaskTrace
from simulator_python.ttd_queue import DynamicTTDAgingQueue, VehicleState

@dataclass
class SubtaskRuntime:
    subtask_id: str
    index: int
    execution_time: int
    status: str = "PENDING"
    max_work_done_pct: float = 0.0

@dataclass
class TaskRuntime:
    task_id: int
    arrival_time: int
    execution_time: int
    deadline: int
    price: int
    packet_id: int = 0
    packet_count: int = 1
    subtasks: List[SubtaskRuntime] = field(default_factory=list)
    current_subtask_idx: int = 0
    current_subtask_start_time: int = 0
    allocated_vu_ids: List[int] = field(default_factory=list)
    recruiter_id: Optional[int] = None
    is_critical: bool = False
    redundancy_n: int = 1

    @property
    def remaining_execution_time(self) -> int:
        remaining = 0
        for st in self.subtasks[self.current_subtask_idx:]:
            remaining += st.execution_time
        return remaining

    def laxity(self, current_time: int) -> int:
        return self.deadline - current_time - self.remaining_execution_time

class SimulationEngine:
    def __init__(self, cars: List[CarTrace], tasks: List[TaskTrace]):
        self.cars_trace = sorted(cars, key=lambda c: c.arrival_time)
        self.tasks_trace = sorted(tasks, key=lambda t: t.arrival_time)
        
        self.ttd_queue = DynamicTTDAgingQueue()
        self.pending_tasks: List[TaskRuntime] = []
        self.active_tasks: List[TaskRuntime] = []
        self.completed_tasks: List[TaskRuntime] = []
        self.failed_tasks: List[TaskRuntime] = []
        
        self.current_time = 0
        self.total_revenue_possible = sum(t.price for t in self.tasks_trace)
        self.total_revenue_earned = 0
        self.total_cost_incurred = 0
        self.task_accepted_count = 0
        self.task_satisfied_count = 0
        
        self._car_idx = 0
        self._task_idx = 0

    def step_time(self, time_step: int = 1):
        self.current_time += time_step

    def process_arrivals(self):
        while self._car_idx < len(self.cars_trace) and self.cars_trace[self._car_idx].arrival_time <= self.current_time:
            c = self.cars_trace[self._car_idx]
            v_state = VehicleState(
                vu_id=c.car_id,
                arrival_time=c.arrival_time,
                departure_time=c.departure_time,
                initial_category=c.car_type
            )
            self.ttd_queue.add_vu(v_state)
            self._car_idx += 1

        while self._task_idx < len(self.tasks_trace) and self.tasks_trace[self._task_idx].arrival_time <= self.current_time:
            t = self.tasks_trace[self._task_idx]
            
            subtask = SubtaskRuntime(
                subtask_id=f"T_{t.task_id}_ST_0",
                index=0,
                execution_time=t.execution_time
            )
            t_runtime = TaskRuntime(
                task_id=t.task_id,
                arrival_time=t.arrival_time,
                execution_time=t.execution_time,
                deadline=t.deadline,
                price=t.price,
                packet_id=t.packet_id,
                packet_count=t.packet_count,
                subtasks=[subtask],
                current_subtask_start_time=self.current_time
            )
            self.pending_tasks.append(t_runtime)
            self._task_idx += 1

    def process_departures(self) -> List[int]:
        departed_vu_ids = []
        for vu_id, vu in list(self.ttd_queue.all_vus.items()):
            if vu.status != "DEPARTED" and vu.departure_time <= self.current_time:
                departed_vu_ids.append(vu_id)
                self.ttd_queue.remove_vu(vu_id)
        return departed_vu_ids

    def calculate_results(self) -> dict:
        net_profit = self.total_revenue_earned - self.total_cost_incurred
        profit_perc = (net_profit / self.total_revenue_possible * 100.0) if self.total_revenue_possible > 0 else 0.0
        return {
            "total_revenue_possible": self.total_revenue_possible,
            "total_revenue_earned": self.total_revenue_earned,
            "total_cost_incurred": self.total_cost_incurred,
            "net_profit": net_profit,
            "profit_percentage": profit_perc,
            "task_accepted": self.task_accepted_count,
            "task_satisfied": self.task_satisfied_count,
            "completed_tasks": len(self.completed_tasks),
            "failed_tasks": len(self.failed_tasks),
        }
