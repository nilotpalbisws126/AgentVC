"""
Judge Agent Module.
Consumes the Debate Transcript from the Task Splitting and VU Allocation Agents
and renders the binding JSON decision using Ollama Llama 3.1 8B/7B.
"""

import json
import logging
from typing import Dict, Any, Optional
from agents.ollama_client import OllamaLLMClient

JUDGE_AGENT_PROMPT_TEMPLATE = """
You are the Judge Agent in a Multi-Agent Vehicular Cloud Orchestrator.
Your role is to evaluate the Debate Transcript between the Task Splitting Agent and VU Allocation Agent
and render a binding, optimal JSON allocation decision.

SYSTEM STATE & REAL IDLE VEHICLE POOL:
- Simulation Time: {current_time}
- Task ID: {task_id}, ExecTime: {exec_time}m, Deadline: {deadline}m, Price: ${price}
- Available LRT Vehicle IDs (TTD > 6h): {sample_lrt_vus} (count: {idle_lrt_count})
- Available MRT Vehicle IDs (3h <= TTD <= 6h): {sample_mrt_vus} (count: {idle_mrt_count})
- Available SRT Vehicle IDs (TTD < 3h): {sample_srt_vus} (count: {idle_srt_count})

DEBATE TRANSCRIPT:
{debate_transcript}

{feedback_prompt}

CRITICAL RULES:
1. DYNAMIC VARIABLE SUBTASKING: Decompose the task into non-equal, variable-sized subtasks matching vehicle stays (e.g. ST1: 60m for SRTs, ST2: 120m for MRTs, ST3: 300m for LRT anchors). Ensure each subtask is >= 45 minutes so VM setup overhead (15m) does not dominate.
2. ADAPTIVE REDUNDANCY LEVEL n (1 to 5): Dynamically decide redundancy n based on task laxity:
   - High Laxity (> 300m): Use lower redundancy n=1 or n=2 to conserve vehicle hours and maximize throughput.
   - Medium Laxity (100m to 300m): Use n=2 or n=3.
   - Tight Laxity (< 100m): Use higher redundancy n=3 or n=4 for MT99R reliability.
3. Select allocated_vu_ids ONLY from the actual available vehicle ID lists above ({sample_lrt_vus}, {sample_mrt_vus}, {sample_srt_vus}). Do NOT output dummy placeholder strings!

INSTRUCTIONS:
1. First, write your step-by-step evaluation in a <scratchpad>...</scratchpad> block analyzing:
   a) Turn 1 vs Turn 2 debate arguments.
   b) Variable subtask length choices vs deadline laxity (min subtask >= 45m).
   c) Adaptive redundancy level n choice.
   d) Real vehicle ID selection from candidate pool.
2. Second, render your binding decision ONLY as a valid JSON block matching this exact schema:

<scratchpad>
Step-by-step reasoning and vehicle evaluation goes here...
</scratchpad>

```json
{{
  "decision_type": "INITIAL_ALLOCATION",
  "task_id": "{task_id}",
  "subtask_decomposition": [
    {{"subtask_index": 1, "duration": 60.0}},
    {{"subtask_index": 2, "duration": 120.0}},
    {{"subtask_index": 3, "duration": 300.0}}
  ],
  "allocated_vu_ids": [3995, 3966],
  "vu_category_mix": {{"LRT": 1, "MRT": 1, "SRT": 0}},
  "initial_redundancy_n": 2,
  "assigned_recruiter_id": 3995,
  "justification": "Approved variable subtasking matching pool stays with adaptive redundancy n=2."
}}
```
"""

class JudgeAgent:
    """Evaluates debate transcript and renders final plan decision via Ollama."""

    def __init__(self, llm_client: Optional[OllamaLLMClient] = None, logger: Optional[logging.Logger] = None):
        self.llm_client = llm_client
        self.logger = logger

    def render_decision(self, system_state: dict, transcript: str, feedback: str = "") -> str:
        sample_lrt = system_state.get("sample_lrt_vus", [])
        sample_mrt = system_state.get("sample_mrt_vus", [])
        sample_srt = system_state.get("sample_srt_vus", [])

        prompt = JUDGE_AGENT_PROMPT_TEMPLATE.format(
            current_time=system_state.get("current_time", 0),
            task_id=system_state.get("task_id", 0),
            exec_time=system_state.get("task_exec_time", 600),
            deadline=system_state.get("deadline", 1200),
            price=system_state.get("price", 100),
            idle_lrt_count=system_state.get("idle_lrt_count", len(sample_lrt)),
            sample_lrt_vus=sample_lrt[:15],
            idle_mrt_count=system_state.get("idle_mrt_count", len(sample_mrt)),
            sample_mrt_vus=sample_mrt[:15],
            idle_srt_count=system_state.get("idle_srt_count", len(sample_srt)),
            sample_srt_vus=sample_srt[:15],
            debate_transcript=transcript,
            feedback_prompt=f"FEEDBACK FROM PREVIOUS ERROR: {feedback}" if feedback else ""
        )

        if self.logger:
            self.logger.info(f"[JudgeAgent] Invoking Judge Agent via Ollama for Task {system_state.get('task_id')}...")

        if self.llm_client:
            try:
                return self.llm_client.query(prompt)
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"[JudgeAgent] Ollama query failed ({e}). Using mock heuristic.")

        l_vus = sample_lrt
        m_vus = sample_mrt
        s_vus = sample_srt

        chosen_vus = []
        if l_vus: chosen_vus.append(l_vus[0])
        if m_vus: chosen_vus.extend(m_vus[:1])
        if not chosen_vus and s_vus: chosen_vus.extend(s_vus[:2])

        exec_t = float(system_state.get("task_exec_time", 600))
        subtasks = []
        if exec_t > 300:
            st1_dur = max(60.0, round(exec_t * 0.15, 1))
            st2_dur = max(90.0, round(exec_t * 0.35, 1))
            st3_dur = max(100.0, round(exec_t * 0.50, 1))
            subtasks = [
                {"subtask_index": 1, "duration": st1_dur},
                {"subtask_index": 2, "duration": st2_dur},
                {"subtask_index": 3, "duration": st3_dur}
            ]
        else:
            subtasks = [{"subtask_index": 1, "duration": exec_t}]

        rec_id = chosen_vus[0] if chosen_vus else None
        laxity = system_state.get("laxity", 300)
        n_rec = 1 if laxity > 300 else (2 if laxity >= 100 else 3)

        return json.dumps({
            "decision_type": "INITIAL_ALLOCATION",
            "task_id": system_state.get("task_id"),
            "subtask_decomposition": subtasks,
            "allocated_vu_ids": chosen_vus[:n_rec],
            "vu_category_mix": {"LRT": len([v for v in chosen_vus[:n_rec] if v in l_vus]), "MRT": len([v for v in chosen_vus[:n_rec] if v in m_vus]), "SRT": len([v for v in chosen_vus[:n_rec] if v in s_vus])},
            "initial_redundancy_n": len(chosen_vus[:n_rec]),
            "assigned_recruiter_id": rec_id,
            "justification": f"Variable subtask allocation with adaptive redundancy n={n_rec}."
        })
