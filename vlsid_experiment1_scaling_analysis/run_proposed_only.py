"""
Standalone Proposed Multi-Agent System Runner.
Executes ONLY the Proposed Agentic Multi-Agent Framework (Debate + TTD Queue + Uncertainty Agent)
on Ollama Llama 3.1 GPU without running prior baselines.
"""

import os
import sys
import logging
import json

# Add parent dir to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from simulator_python.runner_ollama import OllamaBenchmarkRunner

def main():
    print("=========================================================================")
    print("   STARTING STANDALONE PROPOSED MULTI-AGENT SYSTEM BENCHMARK (GPU)")
    print("=========================================================================")
    
    runner = OllamaBenchmarkRunner()
    
    print(f"Loading Car Trace  : {runner.car_trace_path}")
    print(f"Loading Task Trace : {runner.task_trace_path}")
    
    # Run ONLY Proposed Multi-Agent System
    res = runner.run_proposed_multi_agent()
    
    # Save standalone result
    out_results = {"Proposed_Multi_Agent": res}
    out_path = os.path.join(BASE_DIR, "logs", "benchmark_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_results, f, indent=2)
        
    print("\n=========================================================================")
    print("  PROPOSED MULTI-AGENT GPU BENCHMARK COMPLETE!")
    print(f"  Revenue Earned : ${res['total_revenue_earned']:,}")
    print(f"  Cost Incurred  : ${res['total_cost_incurred']:,}")
    print(f"  Net Profit     : ${res['net_profit']:,} ({res['profit_percentage']:.2f}%)")
    print(f"  Satisfied      : {res['task_satisfied']} / {res['task_accepted']} accepted (Failed: {res['failed_tasks']})")
    print(f"  Saved Results  : {out_path}")
    print("=========================================================================")

if __name__ == "__main__":
    main()
