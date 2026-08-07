"""
Dynamic Time-To-Departure (TTD) Aging Queue Module.
Implements dynamic category transitions (LRT -> MRT -> SRT) based on 
the vehicle's actual remaining residency time at time step 't'.
"""

from dataclasses import dataclass
from typing import List, Dict, Set, Optional

@dataclass
class VehicleState:
    vu_id: int
    arrival_time: int
    departure_time: int
    initial_category: str
    status: str = "IDLE"
    assigned_task_id: Optional[int] = None
    assigned_subtask_id: Optional[str] = None
    task_execution_start_time: int = 0
    initial_work_done: float = 0.0
    is_recruiter: bool = False

    def time_to_departure(self, current_time: int) -> int:
        return max(0, self.departure_time - current_time)

    def current_category(self, current_time: int) -> str:
        ttd = self.time_to_departure(current_time)
        if ttd > 360:
            return "L"
        elif ttd >= 180:
            return "M"
        else:
            return "S"

class DynamicTTDAgingQueue:
    def __init__(self):
        self.all_vus: Dict[int, VehicleState] = {}
        self.idle_vus: Set[int] = set()
        self.allocated_vus: Set[int] = set()

    def add_vu(self, vu: VehicleState):
        self.all_vus[vu.vu_id] = vu
        if vu.status == "IDLE":
            self.idle_vus.add(vu.vu_id)

    def remove_vu(self, vu_id: int):
        if vu_id in self.all_vus:
            self.all_vus[vu_id].status = "DEPARTED"
            self.idle_vus.discard(vu_id)
            self.allocated_vus.discard(vu_id)

    def allocate_vu(self, vu_id: int, task_id: int, start_time: int, subtask_id: Optional[str] = None):
        if vu_id in self.idle_vus:
            self.idle_vus.remove(vu_id)
            self.allocated_vus.add(vu_id)
            vu = self.all_vus[vu_id]
            vu.status = "ALLOCATED"
            vu.assigned_task_id = task_id
            vu.assigned_subtask_id = subtask_id
            vu.task_execution_start_time = start_time

    def release_vu(self, vu_id: int):
        if vu_id in self.all_vus:
            vu = self.all_vus[vu_id]
            if vu.status != "DEPARTED":
                vu.status = "IDLE"
                vu.assigned_task_id = None
                vu.assigned_subtask_id = None
                vu.is_recruiter = False
                self.allocated_vus.discard(vu_id)
                self.idle_vus.add(vu_id)

    def get_idle_vus_by_category(self, current_time: int) -> Dict[str, List[VehicleState]]:
        buckets: Dict[str, List[VehicleState]] = {"L": [], "M": [], "S": []}
        for vu_id in list(self.idle_vus):
            vu = self.all_vus[vu_id]
            cat = vu.current_category(current_time)
            buckets[cat].append(vu)
        for cat in buckets:
            buckets[cat].sort(key=lambda v: v.time_to_departure(current_time), reverse=True)
        return buckets

    def get_system_snapshot(self, current_time: int) -> dict:
        buckets = self.get_idle_vus_by_category(current_time)
        return {
            "current_time": current_time,
            "total_idle_vus": len(self.idle_vus),
            "total_allocated_vus": len(self.allocated_vus),
            "idle_lrt_count": len(buckets["L"]),
            "idle_mrt_count": len(buckets["M"]),
            "idle_srt_count": len(buckets["S"]),
            "sample_lrt_vus": [v.vu_id for v in buckets["L"][:15]],
            "sample_mrt_vus": [v.vu_id for v in buckets["M"][:15]],
            "sample_srt_vus": [v.vu_id for v in buckets["S"][:15]],
        }
