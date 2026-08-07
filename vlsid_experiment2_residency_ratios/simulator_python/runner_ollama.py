"""
Ollama GPU Runner Engine for Vehicular Cloud Task Scheduling.
Executes 5-way comparative benchmarks using Ollama API (Llama 3.1 8B/7B) on remote GPU.
Saves all interaction logs, debate transcripts, and results to logs/ folder.

Evaluated Methodologies:
1. Baseline 1: Static Redundant Only (No Checkpointing, MTTF 50% - Florin et al.)
2. Baseline 2: Static Checkpointing Only (No Redundancy n=1 - Ghazizadeh et al.)
3. Baseline 3: Static MT99R + Recruiter Vehicle VU_R (FGCS 2025 SOTA - Sarkar et al.)
4. Baseline 4: Single LLM Agent Baseline (Single LLM.pdf)
5. Proposed System: Agentic Multi-Agent Framework (Debate + TTD Queue + Uncertainty Agent + 15m VM MTTR)
"""

import os
import sys
import json
import time
from typing import Dict, List, Any, Optional

from simulator_python.trace_loader import TraceLoader
from simulator_python.engine import SimulationEngine, TaskRuntime, SubtaskRuntime
from agents.ollama_client import OllamaLLMClient
from agents.logging_config import setup_agent_logger
from agents.sanity_wrapper import SanityValidatorWrapper
from agents.single_llm_agent import SingleLLMAgent
from agents.debate_engine import DebateEngine
from agents.judge_agent import JudgeAgent
from agents.uncertainty_agent import UncertaintyAgent

# Dynamic base directory resolution
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CAR_PATH = os.path.join(BASE_DIR, "Trace_data", "car_data.txt")
DEFAULT_TASK_PATH = os.path.join(BASE_DIR, "Trace_data", "task_data_1000tasks_1000min.txt")
if not os.path.exists(DEFAULT_TASK_PATH):
    DEFAULT_TASK_PATH = os.path.join(BASE_DIR, "Trace_data", "task_data_1000tasks_1000min.txt")

class OllamaBenchmarkRunner:
    def __init__(
        self,
        car_trace_path: Optional[str] = None,
        task_trace_path: Optional[str] = None,
        model_name: str = "llama3.1",
        host_url: str = "http://localhost:11434",
        log_dir: str = "logs",
        cars_override: Optional[List[Any]] = None,
        tasks_override: Optional[List[Any]] = None
    ):
        self.log_dir = log_dir
        self.logger = setup_agent_logger(log_dir=log_dir, log_name="agent_execution.log")
        
        self.car_trace_path = car_trace_path if (car_trace_path and os.path.exists(car_trace_path)) else DEFAULT_CAR_PATH
        self.task_trace_path = task_trace_path if (task_trace_path and os.path.exists(task_trace_path)) else DEFAULT_TASK_PATH

        self.logger.info("Initializing Ollama 5-Way Benchmark Runner for GPU Deployment...")
        self.logger.info(f"Loading Car Trace: {self.car_trace_path}")
        self.logger.info(f"Loading Task Trace: {self.task_trace_path}")

        self.cars = cars_override if cars_override is not None else TraceLoader.load_cars(self.car_trace_path)
        self.tasks = tasks_override if tasks_override is not None else TraceLoader.load_tasks(self.task_trace_path)

        # Initialize Ollama LLM Client
        self.ollama_client = OllamaLLMClient(
            model_name=model_name,
            host_url=host_url,
            logger=self.logger
        )
        self.ollama_client.check_health()

        # Agents
        self.sanity_wrapper = SanityValidatorWrapper(max_retries=2, logger=self.logger)
        self.single_agent = SingleLLMAgent(llm_client=self.ollama_client, logger=self.logger)
        self.debate_engine = DebateEngine(llm_client=self.ollama_client, logger=self.logger)
        self.judge_agent = JudgeAgent(llm_client=self.ollama_client, logger=self.logger)
        self.uncertainty_agent = UncertaintyAgent(llm_client=self.ollama_client, logger=self.logger)

    def _process_llm_allocation(self, engine: SimulationEngine, task: Any, decision: dict, current_time: int):
        alloc_vus = decision.get("allocated_vu_ids", [])
        req_n = decision.get("initial_redundancy_n", len(alloc_vus) if alloc_vus else 2)
        cat_mix = decision.get("vu_category_mix", {})
        
        # Filter integer VU IDs from current idle pool
        valid_alloc_vus = [v for v in alloc_vus if isinstance(v, int) and v in engine.ttd_queue.idle_vus]

        # Process subtask decomposition from LLM if provided
        llm_subtasks = decision.get("subtask_decomposition", [])
        if isinstance(llm_subtasks, list) and len(llm_subtasks) > 0:
            st_objects = []
            for idx, st_dict in enumerate(llm_subtasks):
                dur = float(st_dict.get("duration", task.execution_time))
                st_objects.append(SubtaskRuntime(f"T_{task.task_id}_ST_{idx}", idx, int(dur)))
            task.subtasks = st_objects
            task.current_subtask_idx = 0

        if len(valid_alloc_vus) < req_n:
            buckets = engine.ttd_queue.get_idle_vus_by_category(current_time)
            idle_lrt = [v.vu_id for v in buckets["L"] if v.vu_id not in valid_alloc_vus]
            idle_mrt = [v.vu_id for v in buckets["M"] if v.vu_id not in valid_alloc_vus]
            idle_srt = [v.vu_id for v in buckets["S"] if v.vu_id not in valid_alloc_vus]

            needed = req_n - len(valid_alloc_vus)
            req_l = cat_mix.get("LRT", 0) if isinstance(cat_mix, dict) else 0
            req_m = cat_mix.get("MRT", 0) if isinstance(cat_mix, dict) else 0

            while needed > 0 and req_l > 0 and idle_lrt:
                valid_alloc_vus.append(idle_lrt.pop(0))
                needed -= 1
                req_l -= 1

            while needed > 0 and req_m > 0 and idle_mrt:
                valid_alloc_vus.append(idle_mrt.pop(0))
                needed -= 1
                req_m -= 1

            for pool in [idle_lrt, idle_mrt, idle_srt]:
                while needed > 0 and pool:
                    valid_alloc_vus.append(pool.pop(0))
                    needed -= 1

        if valid_alloc_vus:
            for vid in valid_alloc_vus:
                engine.ttd_queue.allocate_vu(vid, task.task_id, current_time)
            task.allocated_vu_ids = valid_alloc_vus
            task.current_subtask_start_time = current_time
            rec_id = decision.get("assigned_recruiter_id")
            task.recruiter_id = rec_id if (isinstance(rec_id, int) and rec_id in valid_alloc_vus) else valid_alloc_vus[0]
            engine.active_tasks.append(task)
            engine.pending_tasks.remove(task)
            engine.task_accepted_count += 1

    def run_baseline_1_static_redundant(self) -> dict:
        """Baseline 1: Static Redundant Only (No Checkpointing, MTTF 50% - Florin et al.)."""
        self.logger.info("Starting Benchmark 1/5: Baseline 1 Static Redundant (Florin et al.)...")
        engine = SimulationEngine(self.cars, self.tasks)
        max_time = max(c.departure_time for c in engine.cars_trace) if engine.cars_trace else 1000
        
        for t in range(0, max_time + 1, 10):
            engine.current_time = t
            engine.process_arrivals()
            engine.process_departures()
            
            idle_vus = list(engine.ttd_queue.idle_vus)
            for task in list(engine.pending_tasks):
                if len(idle_vus) >= 3 and task.deadline >= t + task.execution_time:
                    chosen = idle_vus[:3]
                    for vid in chosen:
                        engine.ttd_queue.allocate_vu(vid, task.task_id, t)
                    task.allocated_vu_ids = chosen
                    task.recruiter_id = chosen[0]
                    task.current_subtask_start_time = t
                    engine.active_tasks.append(task)
                    engine.pending_tasks.remove(task)
                    engine.task_accepted_count += 1
                    idle_vus = list(engine.ttd_queue.idle_vus)
            
            for task in list(engine.active_tasks):
                active_vus = [vid for vid in task.allocated_vu_ids if engine.ttd_queue.all_vus[vid].status == "ALLOCATED"]
                if not active_vus:
                    engine.active_tasks.remove(task)
                    engine.failed_tasks.append(task)
                    engine.total_cost_incurred += task.execution_time * 3 * 5
                    continue
                
                recruiter = engine.ttd_queue.all_vus[task.recruiter_id]
                work_done = (t - recruiter.task_execution_start_time) / max(1, task.execution_time)
                if work_done >= 1.0:
                    engine.active_tasks.remove(task)
                    if t <= task.deadline:
                        engine.completed_tasks.append(task)
                        engine.total_revenue_earned += task.price
                        engine.task_satisfied_count += 1
                    else:
                        engine.failed_tasks.append(task)
                    engine.total_cost_incurred += task.execution_time * len(task.allocated_vu_ids) * 5
                    for vid in task.allocated_vu_ids:
                        engine.ttd_queue.release_vu(vid)
                        
        res = engine.calculate_results()
        res["algorithm"] = "B1: Static Redundant (Florin et al.)"
        self.logger.info(f"B1 Complete: Net Profit ${res['net_profit']:,} ({res['profit_percentage']:.2f}%)")
        return res

    def run_baseline_2_static_checkpointing(self) -> dict:
        """Baseline 2: Static Checkpointing Only (No Redundancy n=1 - Ghazizadeh et al.)."""
        self.logger.info("Starting Benchmark 2/5: Baseline 2 Static Checkpointing (Ghazizadeh et al.)...")
        engine = SimulationEngine(self.cars, self.tasks)
        max_time = max(c.departure_time for c in engine.cars_trace) if engine.cars_trace else 1000

        for t in range(0, max_time + 1, 10):
            engine.current_time = t
            engine.process_arrivals()
            departed_vids = engine.process_departures()

            for task in list(engine.active_tasks):
                if any(v in departed_vids for v in task.allocated_vu_ids):
                    for vid in task.allocated_vu_ids:
                        if vid in engine.ttd_queue.all_vus:
                            engine.ttd_queue.release_vu(vid)
                    idle_vus = list(engine.ttd_queue.idle_vus)
                    if idle_vus and task.deadline >= t + task.remaining_execution_time:
                        rep = idle_vus[0]
                        engine.ttd_queue.allocate_vu(rep, task.task_id, t)
                        task.allocated_vu_ids = [rep]
                        task.current_subtask_start_time = t
                    else:
                        engine.active_tasks.remove(task)
                        engine.failed_tasks.append(task)

            idle_vus = list(engine.ttd_queue.idle_vus)
            for task in list(engine.pending_tasks):
                if task.execution_time > 300:
                    st1 = SubtaskRuntime(f"T_{task.task_id}_ST_1", 1, int(round(task.execution_time / 2.0)))
                    st2 = SubtaskRuntime(f"T_{task.task_id}_ST_2", 2, int(round(task.execution_time / 2.0)))
                    task.subtasks = [st1, st2]
                if idle_vus and task.deadline >= t + task.execution_time:
                    v = idle_vus.pop(0)
                    engine.ttd_queue.allocate_vu(v, task.task_id, t)
                    task.allocated_vu_ids = [v]
                    task.current_subtask_start_time = t
                    engine.active_tasks.append(task)
                    engine.pending_tasks.remove(task)
                    engine.task_accepted_count += 1

            for task in list(engine.active_tasks):
                curr_st = task.subtasks[task.current_subtask_idx]
                if t - task.current_subtask_start_time >= curr_st.execution_time:
                    if task.current_subtask_idx + 1 < len(task.subtasks):
                        task.current_subtask_idx += 1
                        task.current_subtask_start_time = t
                    else:
                        engine.active_tasks.remove(task)
                        if t <= task.deadline:
                            engine.completed_tasks.append(task)
                            engine.total_revenue_earned += task.price
                            engine.task_satisfied_count += 1
                        else:
                            engine.failed_tasks.append(task)
                        engine.total_cost_incurred += task.execution_time * 1 * 5
                        for vid in task.allocated_vu_ids:
                            engine.ttd_queue.release_vu(vid)

        res = engine.calculate_results()
        res["algorithm"] = "B2: Static Checkpointing Only (Ghazizadeh et al.)"
        self.logger.info(f"B2 Complete: Net Profit ${res['net_profit']:,} ({res['profit_percentage']:.2f}%)")
        return res

    def run_baseline_3_static_mt99r(self) -> dict:
        """Baseline 3: Static MT99R + Recruiter Vehicle VU_R (Sarkar et al. 2025 SOTA)."""
        self.logger.info("Starting Benchmark 3/5: Baseline 3 Static MT99R + Recruiter (Sarkar et al. 2025)...")
        engine = SimulationEngine(self.cars, self.tasks)
        max_time = max(c.departure_time for c in engine.cars_trace) if engine.cars_trace else 1000

        for t in range(0, max_time + 1, 10):
            engine.current_time = t
            engine.process_arrivals()
            departed_vids = engine.process_departures()
            
            for task in list(engine.active_tasks):
                if task.recruiter_id in departed_vids:
                    remaining_vids = [v for v in task.allocated_vu_ids if v not in departed_vids]
                    if remaining_vids:
                        task.recruiter_id = remaining_vids[0]  # AP assigns new Recruiter VU_R
                        idle_vus = list(engine.ttd_queue.idle_vus)
                        if idle_vus and task.deadline >= t + task.remaining_execution_time:
                            rep = idle_vus[0]
                            engine.ttd_queue.allocate_vu(rep, task.task_id, t)
                            task.allocated_vu_ids.append(rep)
                    else:
                        engine.active_tasks.remove(task)
                        engine.failed_tasks.append(task)

            buckets = engine.ttd_queue.get_idle_vus_by_category(t)
            for task in list(engine.pending_tasks):
                l_vus, m_vus, s_vus = buckets["L"], buckets["M"], buckets["S"]
                req_n = 3
                selected = []
                if len(l_vus) >= req_n:
                    selected = [v.vu_id for v in l_vus[:req_n]]
                elif len(m_vus) >= req_n:
                    selected = [v.vu_id for v in m_vus[:req_n]]
                elif len(s_vus) >= req_n:
                    selected = [v.vu_id for v in s_vus[:req_n]]
                    
                if selected and task.deadline >= t + task.execution_time:
                    for vid in selected:
                        engine.ttd_queue.allocate_vu(vid, task.task_id, t)
                    task.allocated_vu_ids = selected
                    task.recruiter_id = selected[0]  # Recruiter Vehicle VU_R
                    task.current_subtask_start_time = t
                    engine.active_tasks.append(task)
                    engine.pending_tasks.remove(task)
                    engine.task_accepted_count += 1
                    buckets = engine.ttd_queue.get_idle_vus_by_category(t)

            for task in list(engine.active_tasks):
                curr_st = task.subtasks[task.current_subtask_idx]
                if t - task.current_subtask_start_time >= curr_st.execution_time:
                    if task.current_subtask_idx + 1 < len(task.subtasks):
                        task.current_subtask_idx += 1
                        task.current_subtask_start_time = t
                    else:
                        engine.active_tasks.remove(task)
                        if t <= task.deadline:
                            engine.completed_tasks.append(task)
                            engine.total_revenue_earned += task.price
                            engine.task_satisfied_count += 1
                        else:
                            engine.failed_tasks.append(task)
                        engine.total_cost_incurred += task.execution_time * len(task.allocated_vu_ids) * 5
                        for vid in task.allocated_vu_ids:
                            engine.ttd_queue.release_vu(vid)

        res = engine.calculate_results()
        res["algorithm"] = "B3: Static MT99R (Sarkar et al. 2025)"
        self.logger.info(f"B3 Complete: Net Profit ${res['net_profit']:,} ({res['profit_percentage']:.2f}%)")
        return res

    def run_baseline_4_single_llm(self) -> dict:
        """Baseline 4: Single LLM Agent Architecture (Single LLM.pdf)."""
        self.logger.info("Starting Benchmark 4/5: Baseline 4 Single LLM Agent via Ollama...")
        engine = SimulationEngine(self.cars, self.tasks)
        max_time = max(c.departure_time for c in engine.cars_trace) if engine.cars_trace else 1000

        for t in range(0, max_time + 1, 10):
            engine.current_time = t
            engine.process_arrivals()
            engine.process_departures()
            
            for task in list(engine.pending_tasks):
                state = engine.ttd_queue.get_system_snapshot(t)
                state.update({
                    "task_id": str(task.task_id),
                    "task_exec_time": task.execution_time,
                    "deadline": task.deadline,
                    "price": task.price,
                    "laxity": task.laxity(t)
                })

                decision = self.sanity_wrapper.run_agent_with_retry_and_fallback(
                    agent_call_fn=lambda s, fb: self.single_agent.generate_decision(s, fb),
                    system_state=state,
                    is_reactive=False
                )

                self._process_llm_allocation(engine, task, decision, t)

            for task in list(engine.active_tasks):
                curr_st = task.subtasks[task.current_subtask_idx]
                if t - task.current_subtask_start_time >= curr_st.execution_time:
                    still_active = [vid for vid in task.allocated_vu_ids if engine.ttd_queue.all_vus[vid].status == "ALLOCATED"]
                    if still_active:
                        if task.current_subtask_idx + 1 < len(task.subtasks):
                            task.current_subtask_idx += 1
                            task.current_subtask_start_time = t
                        else:
                            engine.active_tasks.remove(task)
                            if t <= task.deadline:
                                engine.completed_tasks.append(task)
                                engine.total_revenue_earned += task.price
                                engine.task_satisfied_count += 1
                            else:
                                engine.failed_tasks.append(task)
                            engine.total_cost_incurred += task.execution_time * len(task.allocated_vu_ids) * 5
                            for vid in task.allocated_vu_ids:
                                engine.ttd_queue.release_vu(vid)
                    else:
                        engine.active_tasks.remove(task)
                        engine.failed_tasks.append(task)
                        engine.total_cost_incurred += task.execution_time * len(task.allocated_vu_ids) * 5
                        for vid in task.allocated_vu_ids:
                            engine.ttd_queue.release_vu(vid)

        res = engine.calculate_results()
        res["algorithm"] = "B4: Single LLM Agent Baseline (Single LLM.pdf)"
        self.logger.info(f"B4 Complete: Net Profit ${res['net_profit']:,} ({res['profit_percentage']:.2f}%)")
        return res

    def run_proposed_multi_agent(self) -> dict:
        """Proposed System: Agentic Multi-Agent Framework with Dynamic TTD Queue & Uncertainty Agent."""
        self.logger.info("Starting Benchmark 5/5: Proposed Multi-Agent System via Ollama...")
        engine = SimulationEngine(self.cars, self.tasks)
        max_time = max(c.departure_time for c in engine.cars_trace) if engine.cars_trace else 1000

        for t in range(0, max_time + 1, 10):
            engine.current_time = t
            engine.process_arrivals()
            departed_vids = engine.process_departures()
            
            # 1. Reactive Uncertainty Path (multi agent 2.pdf)
            for task in list(engine.active_tasks):
                departed_in_task = [v for v in task.allocated_vu_ids if v in departed_vids]
                if departed_in_task:
                    u_state = engine.ttd_queue.get_system_snapshot(t)
                    task_laxity = task.laxity(t)
                    vm_delay = 15  # 15 minutes VM provisioning & image installation MTTR penalty
                    effective_laxity = task_laxity - vm_delay
                    
                    u_state.update({
                        "event_type": "EARLY_VEHICLE_DEPARTURE",
                        "departed_vu_id": str(departed_in_task[0]),
                        "task_id": str(task.task_id),
                        "subtask_id": f"ST_{task.current_subtask_idx}",
                        "rem_exec_time": task.remaining_execution_time,
                        "rem_deadline": task.deadline - t,
                        "laxity": task_laxity,
                        "vm_delay_min": vm_delay,
                        "effective_laxity": effective_laxity,
                        "recruiter_id": task.recruiter_id
                    })
                    
                    decision = self.sanity_wrapper.run_agent_with_retry_and_fallback(
                        agent_call_fn=lambda s, fb: self.uncertainty_agent.handle_uncertainty_event(s, fb),
                        system_state=u_state,
                        is_reactive=True
                    )
                    
                    if decision.get("task_aborted"):
                        engine.active_tasks.remove(task)
                        engine.failed_tasks.append(task)
                        for vid in task.allocated_vu_ids:
                            engine.ttd_queue.release_vu(vid)
                    else:
                        rep_id = decision.get("replacement_vu_id")
                        if isinstance(rep_id, int) and rep_id in engine.ttd_queue.idle_vus:
                            engine.ttd_queue.allocate_vu(rep_id, task.task_id, t)
                            task.allocated_vu_ids.append(rep_id)

            # 2. Plan-Time Admission Path (Task Arrival -> Debate -> Judge Agent)
            for task in list(engine.pending_tasks):
                p_state = engine.ttd_queue.get_system_snapshot(t)
                p_state.update({
                    "current_time": t,
                    "task_id": str(task.task_id),
                    "task_exec_time": task.execution_time,
                    "deadline": task.deadline,
                    "price": task.price,
                    "laxity": task.laxity(t)
                })

                debate_transcript = self.debate_engine.run_debate(p_state)

                decision = self.sanity_wrapper.run_agent_with_retry_and_fallback(
                    agent_call_fn=lambda s, fb: self.judge_agent.render_decision(s, debate_transcript, fb),
                    system_state=p_state,
                    is_reactive=False
                )

                self._process_llm_allocation(engine, task, decision, t)

            for task in list(engine.active_tasks):
                curr_st = task.subtasks[task.current_subtask_idx]
                if t - task.current_subtask_start_time >= curr_st.execution_time:
                    still_active = [vid for vid in task.allocated_vu_ids if engine.ttd_queue.all_vus[vid].status == "ALLOCATED"]
                    if still_active:
                        if task.current_subtask_idx + 1 < len(task.subtasks):
                            task.current_subtask_idx += 1
                            task.current_subtask_start_time = t
                        else:
                            engine.active_tasks.remove(task)
                            if t <= task.deadline:
                                engine.completed_tasks.append(task)
                                engine.total_revenue_earned += task.price
                                engine.task_satisfied_count += 1
                            else:
                                engine.failed_tasks.append(task)
                            engine.total_cost_incurred += task.execution_time * len(task.allocated_vu_ids) * 5
                            for vid in task.allocated_vu_ids:
                                engine.ttd_queue.release_vu(vid)
                    else:
                        engine.active_tasks.remove(task)
                        engine.failed_tasks.append(task)
                        engine.total_cost_incurred += task.execution_time * len(task.allocated_vu_ids) * 5
                        for vid in task.allocated_vu_ids:
                            engine.ttd_queue.release_vu(vid)

        res = engine.calculate_results()
        res["algorithm"] = "Proposed: Agentic Multi-Agent Framework (Debate + TTD Queue + Uncertainty Agent)"
        self.logger.info(f"Proposed Multi-Agent Complete: Net Profit ${res['net_profit']:,} ({res['profit_percentage']:.2f}%)")
        return res

    def run_all_benchmarks(self) -> dict:
        results = {}
        log_directory = getattr(self, 'log_dir', 'logs')
        os.makedirs(log_directory, exist_ok=True)
        results_path = os.path.join(log_directory, "benchmark_results.json")

        results["B1_Static_Redundant"] = self.run_baseline_1_static_redundant()
        with open(results_path, "w", encoding="utf-8") as f: json.dump(results, f, indent=2)

        results["B2_Static_Checkpointing"] = self.run_baseline_2_static_checkpointing()
        with open(results_path, "w", encoding="utf-8") as f: json.dump(results, f, indent=2)

        results["B3_Static_MT99R"] = self.run_baseline_3_static_mt99r()
        with open(results_path, "w", encoding="utf-8") as f: json.dump(results, f, indent=2)

        results["B4_Single_LLM"] = self.run_baseline_4_single_llm()
        with open(results_path, "w", encoding="utf-8") as f: json.dump(results, f, indent=2)

        results["Proposed_Multi_Agent"] = self.run_proposed_multi_agent()
        with open(results_path, "w", encoding="utf-8") as f: json.dump(results, f, indent=2)

        self.logger.info(f"All 5 benchmarks complete! Results saved to {results_path}")
        return results

if __name__ == "__main__":
    car_path = sys.argv[1] if (len(sys.argv) > 1 and os.path.exists(sys.argv[1])) else DEFAULT_CAR_PATH
    task_path = sys.argv[2] if (len(sys.argv) > 2 and os.path.exists(sys.argv[2])) else DEFAULT_TASK_PATH

    runner = OllamaBenchmarkRunner(
        car_trace_path=car_path,
        task_trace_path=task_path
    )
    runner.run_all_benchmarks()
