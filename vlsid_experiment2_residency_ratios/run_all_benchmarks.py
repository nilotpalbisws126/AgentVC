"""
Full Benchmark Suite Runner for Static Vehicular Cloud Systems.
Executes all 5 Comparative Methodologies:
1. Baseline 1: Static Redundant Only (Florin et al.)
2. Baseline 2: Static Checkpointing Only (Ghazizadeh et al.)
3. Baseline 3: Static MT99R SOTA (Sarkar et al. 2025)
4. Baseline 4: Single LLM Agent Baseline
5. Proposed: Agentic AI Multi-Agent Framework (Debate + TTD Queue + Uncertainty Agent)
"""

import os
import sys
import json
import logging

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from simulator_python.runner_ollama import OllamaBenchmarkRunner

def main():
    print("=========================================================================")
    print("   STARTING FULL 5-BENCHMARK COMPARATIVE SUITE (OLLAMA GPU)")
    print("=========================================================================")
    
    runner = OllamaBenchmarkRunner()
    
    print(f"Car Trace Path  : {runner.car_trace_path}")
    print(f"Task Trace Path : {runner.task_trace_path}")
    print("-------------------------------------------------------------------------")
    
    # Run all benchmarks
    results = runner.run_all_benchmarks()
    
    out_path = os.path.join(BASE_DIR, "logs", "full_benchmark_results.json")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
        
    print("\n=========================================================================")
    print("  FULL BENCHMARK SUITE COMPLETE!")
    print(f"  Saved Results JSON : {out_path}")
    print("=========================================================================")
    
    for b_name, b_res in results.items():
        if isinstance(b_res, dict) and "net_profit" in b_res:
            print(f"  {b_name:30s} | Revenue: ${b_res['total_revenue_earned']:>10,} | Profit: ${b_res['net_profit']:>10,} ({b_res['profit_percentage']:>6.2f}%)")

if __name__ == "__main__":
    main()
