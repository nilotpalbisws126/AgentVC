"""
Reactive Uncertainty Agent Module (multi agent 2.pdf).
Handles runtime events (early vehicle departure, subtask failure, laxity drop)
via fast Ollama Llama 3.1 8B/7B inference on remote GPU.
Includes explicit 15-minute VM installation/provisioning overhead (MTTR = 15 min).
"""

import json
import logging
from typing import Dict, Any, Optional
from agents.ollama_client import OllamaLLMClient

UNCERTAINTY_AGENT_PROMPT_TEMPLATE = """
You are the specialized Reactive Uncertainty Agent in a Multi-Agent Vehicular Cloud System.
Your job is to respond to runtime exception events (car leaving, subtask failure) with minimal latency.

RUNTIME EVENT TRIGGER:
- Event: {event_type}
- Departed / Failed VU: {departed_vu_id}
- Task ID: {task_id}, Subtask ID: {subtask_id}
- Remaining Subtask Execution Time: {rem_exec_time} min
- Remaining Task Deadline: {rem_deadline} min
- Task Laxity: {laxity} min
- VM Provisioning & Image Installation Delay (MTTR): {vm_delay_min} min
- Effective Laxity (after 15 min VM setup): {effective_laxity} min

DYNAMIC TTD QUEUE STATE (ALL ACTIVE & IDLE VUs IN AZ):
- Available LRT VUs (TTD > 6h): {idle_lrt_count} (IDs: {sample_lrt_vus})
- Available MRT VUs (3h <= TTD <= 6h): {idle_mrt_count} (IDs: {sample_mrt_vus})
- Available SRT VUs (TTD < 3h): {idle_srt_count} (IDs: {sample_srt_vus})

INSTRUCTIONS:
1. Note that installing the VM image on a new replacement vehicle takes exactly {vm_delay_min} minutes.
2. First, write your step-by-step evaluation in a <scratchpad>...</scratchpad> block analyzing:
   a) Vehicle departure event and remaining execution time.
   b) 15-min VM setup penalty vs Effective Laxity ({effective_laxity} min).
   c) Decision to abort (if Effective Laxity < 0) or replace from idle pool ({sample_lrt_vus}, {sample_mrt_vus}, {sample_srt_vus}).
3. Second, render your decision ONLY as a valid JSON block matching this exact schema:

<scratchpad>
Step-by-step reasoning and effective laxity evaluation goes here...
</scratchpad>

```json
{{
  "decision_type": "UNCERTAINTY_MITIGATION",
  "event_triggered": "{event_type}",
  "departed_vu_id": "{departed_vu_id}",
  "task_id": "{task_id}",
  "subtask_id": "{subtask_id}",
  "action": "REPLACE_AND_ADJUST_REDUNDANCY",
  "replacement_vu_id": "VU_REPLACEMENT_ID",
  "new_redundancy_n": 3,
  "copy_progress_from_vu_id": "VU_RECRUITER_ID",
  "task_aborted": false,
  "vm_installation_overhead_min": {vm_delay_min},
  "justification": "Replaced departed VU with 15-min VM setup. Effective laxity >= 0 ensures deadline compliance."
}}
```
"""

class UncertaintyAgent:
    """Reactive Uncertainty Agent for runtime dynamic exception handling via Ollama."""

    def __init__(self, llm_client: Optional[OllamaLLMClient] = None, logger: Optional[logging.Logger] = None):
        self.llm_client = llm_client
        self.logger = logger

    def handle_uncertainty_event(self, system_state: dict, feedback: str = "") -> str:
        laxity = system_state.get("laxity", 300)
        vm_delay = system_state.get("vm_delay_min", 15)
        effective_laxity = system_state.get("effective_laxity", laxity - vm_delay)

        prompt = UNCERTAINTY_AGENT_PROMPT_TEMPLATE.format(
            event_type=system_state.get("event_type", "EARLY_VEHICLE_DEPARTURE"),
            departed_vu_id=system_state.get("departed_vu_id", "VU_UNKNOWN"),
            task_id=system_state.get("task_id", "T_UNKNOWN"),
            subtask_id=system_state.get("subtask_id", "ST_UNKNOWN"),
            rem_exec_time=system_state.get("rem_exec_time", 300),
            rem_deadline=system_state.get("rem_deadline", 600),
            laxity=laxity,
            vm_delay_min=vm_delay,
            effective_laxity=effective_laxity,
            idle_lrt_count=system_state.get("idle_lrt_count", 0),
            sample_lrt_vus=system_state.get("sample_lrt_vus", [])[:15],
            idle_mrt_count=system_state.get("idle_mrt_count", 0),
            sample_mrt_vus=system_state.get("sample_mrt_vus", [])[:15],
            idle_srt_count=system_state.get("idle_srt_count", 0),
            sample_srt_vus=system_state.get("sample_srt_vus", [])[:15],
            feedback_prompt=f"FEEDBACK FROM PREVIOUS ERROR: {feedback}" if feedback else ""
        )

        if self.logger:
            self.logger.info(f"[UncertaintyAgent] Handling event {system_state.get('event_type')} for Task {system_state.get('task_id')} (VM Setup: {vm_delay} min, Effective Laxity: {effective_laxity} min)")

        def get_fallback_rep_id():
            for pool in [system_state.get("sample_lrt_vus"), system_state.get("sample_mrt_vus"), system_state.get("sample_srt_vus")]:
                if pool and len(pool) > 0:
                    return pool[0]
            return None

        if self.llm_client:
            try:
                raw_response = self.llm_client.query(prompt, system_prompt="You are a reactive exception handling agent.")
                return raw_response
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"[UncertaintyAgent] Ollama query failed ({e}). Executing dynamic fallback decision.")

        is_aborted = effective_laxity < 0
        rep_id = None if is_aborted else get_fallback_rep_id()
        action = "ABORT_TASK" if is_aborted else "REPLACE_AND_ADJUST_REDUNDANCY"
        justification = (f"Effective laxity ({effective_laxity}m) < 0 due to 15m VM setup delay. Aborted to avoid waste."
                         if is_aborted else f"Replaced departed VU with VU {rep_id}. Effective laxity ({effective_laxity}m) >= 0.")

        return json.dumps({
            "decision_type": "UNCERTAINTY_MITIGATION",
            "event_triggered": system_state.get("event_type", "EARLY_VEHICLE_DEPARTURE"),
            "departed_vu_id": system_state.get("departed_vu_id"),
            "task_id": system_state.get("task_id"),
            "subtask_id": system_state.get("subtask_id"),
            "action": action,
            "replacement_vu_id": rep_id,
            "new_redundancy_n": 3,
            "copy_progress_from_vu_id": system_state.get("recruiter_id"),
            "task_aborted": is_aborted,
            "vm_installation_overhead_min": vm_delay,
            "justification": justification
        })
