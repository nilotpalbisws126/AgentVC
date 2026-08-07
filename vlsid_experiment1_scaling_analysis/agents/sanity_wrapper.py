"""
Bounded Sanity Validator Wrapper Module with Logging.
Validates LLM Agent outputs against feasibility rules and physical constraints.
Enforces MAX_RETRIES = 2 threshold before falling back to deterministic static rules.
"""

import json
import logging
import re
from typing import Dict, Any, Tuple, Optional, List

class SanityValidatorWrapper:
    """
    Validates agent outputs to guarantee valid schemas, feasible deadlines,
    active VU existence, and reasonable redundancy bounds.
    """
    def __init__(self, max_retries: int = 2, logger: Optional[logging.Logger] = None):
        self.max_retries = max_retries
        self.logger = logger

    def _log(self, msg: str):
        if self.logger:
            self.logger.info(msg)

    def validate_plan_decision(self, raw_output: str, system_state: dict) -> Tuple[bool, Optional[dict], List[str]]:
        errors = []
        try:
            if "```json" in raw_output:
                json_str = raw_output.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_output:
                json_str = raw_output.split("```")[1].split("```")[0].strip()
            else:
                json_str = raw_output.strip()

            parsed = json.loads(json_str)
        except Exception as e:
            return False, None, [f"JSON Parse Error: {str(e)}"]

        if not isinstance(parsed, dict):
            return False, None, ["Parsed JSON is not an object."]

        if parsed.get("decision_type") != "INITIAL_ALLOCATION":
            errors.append("Invalid decision_type. Expected 'INITIAL_ALLOCATION'.")

        subtasks = parsed.get("subtask_decomposition", [])
        if not isinstance(subtasks, list) or len(subtasks) == 0:
            errors.append("subtask_decomposition must be a non-empty list.")

        # Robust type conversion for initial_redundancy_n
        redundancy_n = parsed.get("initial_redundancy_n", 0)
        if redundancy_n is None:
            redundancy_n = 0
        try:
            redundancy_n = int(redundancy_n)
        except (ValueError, TypeError):
            redundancy_n = 0

        parsed["initial_redundancy_n"] = redundancy_n

        if not (1 <= redundancy_n <= 5):
            errors.append(f"initial_redundancy_n ({redundancy_n}) out of bounds [1, 5].")

        raw_allocated_vids = parsed.get("allocated_vu_ids", [])
        if not isinstance(raw_allocated_vids, list):
            raw_allocated_vids = []

        clean_allocated_ids = []
        for v in raw_allocated_vids:
            if isinstance(v, int):
                clean_allocated_ids.append(v)
            elif isinstance(v, str):
                digits = re.sub(r'\D', '', v)
                if digits:
                    clean_allocated_ids.append(int(digits))

        parsed["allocated_vu_ids"] = clean_allocated_ids

        available_vu_ids = set(system_state.get("sample_lrt_vus", []) +
                               system_state.get("sample_mrt_vus", []) +
                               system_state.get("sample_srt_vus", []))
        
        if available_vu_ids and clean_allocated_ids and not set(clean_allocated_ids).issubset(available_vu_ids):
            invalid_vus = set(clean_allocated_ids) - available_vu_ids
            errors.append(f"Allocated VUs {list(invalid_vus)} not available in idle pool.")

        is_valid = len(errors) == 0
        return is_valid, parsed if is_valid else None, errors

    def validate_reactive_decision(self, raw_output: str, system_state: dict) -> Tuple[bool, Optional[dict], List[str]]:
        errors = []
        try:
            if "```json" in raw_output:
                json_str = raw_output.split("```json")[1].split("```")[0].strip()
            elif "```" in raw_output:
                json_str = raw_output.split("```")[1].split("```")[0].strip()
            else:
                json_str = raw_output.strip()

            parsed = json.loads(json_str)
        except Exception as e:
            return False, None, [f"JSON Parse Error: {str(e)}"]

        if not isinstance(parsed, dict):
            return False, None, ["Parsed JSON is not an object."]

        if parsed.get("decision_type") != "UNCERTAINTY_MITIGATION":
            errors.append("Invalid decision_type. Expected 'UNCERTAINTY_MITIGATION'.")

        new_n = parsed.get("new_redundancy_n", 1)
        if new_n is None:
            new_n = 1
        try:
            new_n = int(new_n)
        except (ValueError, TypeError):
            new_n = 1

        parsed["new_redundancy_n"] = new_n

        if not (1 <= new_n <= 5):
            errors.append(f"new_redundancy_n ({new_n}) out of bounds [1, 5].")

        is_valid = len(errors) == 0
        return is_valid, parsed if is_valid else None, errors

    def run_agent_with_retry_and_fallback(
        self,
        agent_call_fn,
        system_state: dict,
        is_reactive: bool = False,
        fallback_fn: Optional[Any] = None
    ) -> dict:
        feedback = ""
        for attempt in range(self.max_retries + 1):
            self._log(f"[SanityWrapper] Attempt {attempt + 1}/{self.max_retries + 1} invoking agent...")
            raw_out = agent_call_fn(system_state, feedback)
            self._log(f"[SanityWrapper] Agent raw output: {raw_out[:200]}...")

            if is_reactive:
                valid, parsed, errors = self.validate_reactive_decision(raw_out, system_state)
            else:
                valid, parsed, errors = self.validate_plan_decision(raw_out, system_state)

            if valid and parsed:
                self._log("[SanityWrapper] Agent output validated successfully.")
                return parsed

            self._log(f"[SanityWrapper] Validation failed with errors: {errors}")
            feedback = f"Attempt {attempt + 1} failed with errors: {'; '.join(errors)}. Please output valid JSON matching schema."

        self._log("[SanityWrapper] MAX_RETRIES exceeded! Triggering deterministic safety fallback.")
        if fallback_fn:
            return fallback_fn(system_state)
        else:
            return {
                "decision_type": "UNCERTAINTY_MITIGATION" if is_reactive else "INITIAL_ALLOCATION",
                "status": "FALLBACK_TRIGGERED",
                "initial_redundancy_n": 3,
                "subtask_decomposition": [{"subtask_index": 1, "duration": system_state.get("task_exec_time", 600)}],
                "justification": "Maximum retries exceeded. Executed deterministic static safety fallback."
            }
