"""
Multi-Agent Debate Engine Module.
Coordinates Task Selection, Task Splitting, and VU Allocation Agents.
Conducts multi-turn debates using Ollama Llama 3.1 8B/7B model on GPU.
"""

import json
import logging
import os
from typing import Dict, Any, List, Optional
from agents.ollama_client import OllamaLLMClient

TASK_SPLITTING_AGENT_PROMPT = """
You are the Task Splitting Agent in a Multi-Agent Vehicular Cloud Orchestrator.
Your sole focus is TASK RELIABILITY and DEADLINE FEASIBILITY.

TASK CONSTRAINTS:
- Task ID: {task_id}
- Total Execution Time: {exec_time} min
- Deadline: {deadline} min
- Laxity: {laxity} min

INSTRUCTIONS:
1. Advocate for dynamic, variable-sized subtask decomposition matching vehicle stay durations (e.g. ST1: 60m for quick SRT execution, ST2: 120m for MRT execution, ST3: 300m for LRT anchors). Ensure subtasks are >= 45 minutes so VM setup overhead (15m) does not dominate.
2. Dynamically choose initial redundancy level n (1 to 5) based on task laxity:
   - High Laxity (> 300m): Propose lower redundancy n=1 or n=2 to conserve vehicle capacity.
   - Tight Laxity (< 100m): Propose higher redundancy n=3 or n=4 to guarantee MT99R deadline compliance.
3. Provide a clear 2-sentence rationale prioritizing task completion success.
"""

VU_ALLOCATION_AGENT_PROMPT = """
You are the VU Allocation Agent in a Multi-Agent Vehicular Cloud Orchestrator.
Your sole focus is RESOURCE EFFICIENCY and CONSERVING SCARCE LRT VEHICLES.

IDLE VEHICLE POOL (DYNAMIC TTD QUEUE):
- Available LRT Vehicle IDs (TTD > 6h): {sample_lrt_vus} (count: {idle_lrt_count})
- Available MRT Vehicle IDs (3h <= TTD <= 6h): {sample_mrt_vus} (count: {idle_mrt_count})
- Available SRT Vehicle IDs (TTD < 3h): {sample_srt_vus} (count: {idle_srt_count})

PROPOSAL FROM TASK SPLITTING AGENT:
{splitting_proposal}

INSTRUCTIONS:
Evaluate the Task Splitting Agent's proposal against current VU pool availability.
Advocate for heterogeneous mixing: recruit idle SRT/MRT vehicles for short subtasks (60m-120m) to preserve scarce LRT anchor vehicles for long subtasks (200m+).
If laxity is high, argue for lower redundancy n=1 or n=2 to save vehicle hours and maximize throughput.
Propose specific VU assignments from the real available IDs above.
"""

class TaskSelectionAgent:
    """Selects and prioritizes tasks for admission."""
    def rank_tasks(self, tasks: List[dict]) -> List[dict]:
        return sorted(tasks, key=lambda t: (t["price"] / max(1, t["task_exec_time"]), -t["laxity"]), reverse=True)

class TaskSplittingAgent:
    """Advocates for subtask granularity and reliability."""
    def generate_proposal(self, system_state: dict, llm_client: Optional[OllamaLLMClient] = None) -> str:
        prompt = TASK_SPLITTING_AGENT_PROMPT.format(
            task_id=system_state.get("task_id"),
            exec_time=system_state.get("task_exec_time"),
            deadline=system_state.get("deadline"),
            laxity=system_state.get("laxity")
        )
        if llm_client:
            try:
                return llm_client.query(prompt)
            except Exception:
                pass
        exec_t = system_state.get("task_exec_time", 600)
        laxity = system_state.get("laxity", 300)
        n_rec = 1 if laxity > 300 else (2 if laxity >= 100 else 3)
        return f"PROPOSAL: Split Task into dynamic subtasks (60m, 120m, 300m) matching vehicle pool stays. Request adaptive redundancy n={n_rec} to balance reliability and capacity."

class VUAllocationAgent:
    """Advocates for heterogeneous VU mixing and pool conservation."""
    def generate_counter_proposal(self, system_state: dict, splitting_proposal: str, llm_client: Optional[OllamaLLMClient] = None) -> str:
        sample_lrt = system_state.get("sample_lrt_vus", [])
        sample_mrt = system_state.get("sample_mrt_vus", [])
        sample_srt = system_state.get("sample_srt_vus", [])

        prompt = VU_ALLOCATION_AGENT_PROMPT.format(
            idle_lrt_count=system_state.get("idle_lrt_count", len(sample_lrt)),
            sample_lrt_vus=sample_lrt[:15],
            idle_mrt_count=system_state.get("idle_mrt_count", len(sample_mrt)),
            sample_mrt_vus=sample_mrt[:15],
            idle_srt_count=system_state.get("idle_srt_count", len(sample_srt)),
            sample_srt_vus=sample_srt[:15],
            splitting_proposal=splitting_proposal
        )
        if llm_client:
            try:
                return llm_client.query(prompt)
            except Exception:
                pass
        return "COUNTER-PROPOSAL: Accept dynamic subtasking. Allocate heterogeneous mix (1 LRT anchor + MRT/SRT VUs) from available pool to conserve LRT capacity."

class DebateEngine:
    """Coordinates turn-taking debate between agents and logs debate transcripts."""
    def __init__(self, llm_client: Optional[OllamaLLMClient] = None, logger: Optional[logging.Logger] = None, transcript_log_path: str = "logs/debate_transcripts.jsonl"):
        self.llm_client = llm_client
        self.logger = logger
        self.transcript_log_path = transcript_log_path
        self.splitting_agent = TaskSplittingAgent()
        self.allocation_agent = VUAllocationAgent()
        os.makedirs(os.path.dirname(self.transcript_log_path), exist_ok=True)

    def run_debate(self, system_state: dict) -> str:
        return self.conduct_debate(system_state)

    def conduct_debate(self, system_state: dict) -> str:
        t_id = system_state.get("task_id")
        if self.logger:
            self.logger.info(f"[DebateEngine] Initiating debate for Task {t_id}...")

        prop = self.splitting_agent.generate_proposal(system_state, self.llm_client)
        counter_prop = self.allocation_agent.generate_counter_proposal(system_state, prop, self.llm_client)

        transcript = (
            f"=== DEBATE TRANSCRIPT FOR TASK {t_id} ===\n"
            f"[Turn 1 - Task Splitting Agent (Reliability Advocate)]:\n"
            f"Rationale: Advocating for dynamic variable-sized subtasks and adaptive redundancy n.\n"
            f"Proposal: {prop}\n\n"
            f"[Turn 2 - VU Allocation Agent (Resource Efficiency Advocate)]:\n"
            f"Rationale: Counter-advocating for LRT pool conservation using dynamic SRT/MRT mixing.\n"
            f"Counter-Proposal: {counter_prop}\n"
            f"================================================"
        )

        try:
            record = {
                "task_id": t_id,
                "current_time": system_state.get("current_time"),
                "task_exec_time": system_state.get("task_exec_time"),
                "laxity": system_state.get("laxity"),
                "splitting_agent_proposal": prop,
                "allocation_agent_counter_proposal": counter_prop,
                "full_transcript": transcript
            }
            with open(self.transcript_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            if self.logger:
                self.logger.warning(f"[DebateEngine] Failed to write transcript log: {e}")

        if self.logger:
            self.logger.info(f"[DebateEngine] Multi-Agent Debate for Task {t_id} completed.")
            self.logger.info(f"[DebateEngine] Transcript & Decision Reason:\n{transcript}")

        return transcript
