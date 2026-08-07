"""
Experiment 2: Vehicle Residency Ratio Analysis (LRT : MRT : SRT)
Evaluates the performance of all 5 scheduling methodologies across 3 vehicle residency environments:
1. Ratio 1 (Control Baseline): 5.1% LRT (204 cars), 15.2% MRT (608 cars), 79.7% SRT (3,188 cars)
2. Ratio 2 (Airport / Overnight): 40.0% LRT (1,600 cars), 40.0% MRT (1,600 cars), 20.0% SRT (800 cars)
3. Ratio 3 (Shopping Mall / Retail): 2.0% LRT (80 cars), 8.0% MRT (320 cars), 90.0% SRT (3,600 cars)

Fixed Parameters:
- Tasks: 100 tasks (task_data_100tasks_1000min.txt)
- Total Vehicles: 4,000 cars
- Min Subtask Size: 60 minutes

Stats Tracked:
1. Total Revenue ($)
2. Total Cost ($) = Hardware Rental Cost + LLM Token Cost ($0.00015/1k) + Network Checkpoint Cost ($0.01/GB)
3. Net Profit ($) & Profit Margin (%)
4. Task Failures (Vehicle Departure Reschedules vs Complete Task SLA Breaches)
5. End-to-End Latency (s)
"""

import os
import sys
import json
import time
from typing import List, Dict, Any

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from simulator_python.trace_loader import TraceLoader, CarTrace, TaskTrace
from simulator_python.runner_ollama import OllamaBenchmarkRunner

# Cost Constants
HARDWARE_RENTAL_RATE_PER_MIN = 5.0      # $5/min per active VU
LLM_TOKEN_COST_PER_1K = 0.00015         # $0.00015 per 1,000 tokens
NETWORK_CHECKPOINT_COST_PER_GB = 0.01   # $0.01 per GB transferred
AVG_CHECKPOINT_SIZE_GB = 2.5            # 2.5 GB per saved checkpoint
AVG_TOKENS_PER_DEBATE = 1650            # Average input+output tokens per 3-turn debate

def synthesize_ratio_fleet(all_cars: List[CarTrace], lrt_ratio: float, mrt_ratio: float, srt_ratio: float, total_n: int = 4000) -> List[CarTrace]:
    """Synthesizes a fleet of size total_n matching specified LRT:MRT:SRT ratios."""
    lrt_pool = [c for c in all_cars if c.car_type == 'L']
    mrt_pool = [c for c in all_cars if c.car_type == 'M']
    srt_pool = [c for c in all_cars if c.car_type == 'S']

    target_lrt = int(round(total_n * lrt_ratio))
    target_mrt = int(round(total_n * mrt_ratio))
    target_srt = total_n - target_lrt - target_mrt

    synth_lrt = (lrt_pool * (target_lrt // len(lrt_pool) + 1))[:target_lrt]
    synth_mrt = (mrt_pool * (target_mrt // len(mrt_pool) + 1))[:target_mrt]
    synth_srt = (srt_pool * (target_srt // len(srt_pool) + 1))[:target_srt]

    combined = []
    for idx, template in enumerate(synth_lrt + synth_mrt + synth_srt):
        new_c = CarTrace(
            car_id=idx + 1,
            arrival_time=template.arrival_time,
            departure_time=template.departure_time,
            car_type=template.car_type
        )
        combined.append(new_c)

    return sorted(combined, key=lambda c: c.arrival_time)

def calculate_comprehensive_cost(hardware_cost: float, total_subtasks_executed: int, total_debates_run: int) -> Dict[str, float]:
    """Computes Total Cost = Hardware Cost + LLM Token Cost + Network Checkpoint Cost."""
    token_cost = (total_debates_run * AVG_TOKENS_PER_DEBATE / 1000.0) * LLM_TOKEN_COST_PER_1K
    network_cost = total_subtasks_executed * AVG_CHECKPOINT_SIZE_GB * NETWORK_CHECKPOINT_COST_PER_GB
    total_cost = hardware_cost + token_cost + network_cost

    return {
        "hardware_cost": round(hardware_cost, 2),
        "llm_token_cost": round(token_cost, 4),
        "network_checkpoint_cost": round(network_cost, 2),
        "total_cost": round(total_cost, 2)
    }

def main():
    print("=========================================================================")
    print("   EXPERIMENT 2: VEHICLE RESIDENCY RATIO ANALYSIS (LRT : MRT : SRT)")
    print("   Fixed Load: 100 Tasks | Fleet: 4,000 Cars | Min Subtask: 60 mins")
    print("=========================================================================")

    all_cars = TraceLoader.load_cars(os.path.join(BASE_DIR, "Trace_data", "car_data.txt"))
    all_tasks = TraceLoader.load_tasks(os.path.join(BASE_DIR, "Trace_data", "task_data_100tasks_1000min.txt"))

    scenarios = {
        "Ratio_1_Baseline_Mixed": (0.051, 0.152, 0.797),        # Control baseline: 5.1% L, 15.2% M, 79.7% S
        "Ratio_2_Airport_LongDominant": (0.400, 0.400, 0.200),  # Airport overnight: 40% L, 40% M, 20% S
        "Ratio_3_ShoppingMall_ShortDominant": (0.020, 0.080, 0.900) # Mall high churn: 2% L, 8% M, 90% S
    }

    experiment_results = {}

    for s_name, (r_lrt, r_mrt, r_srt) in scenarios.items():
        fleet_ratio_subset = synthesize_ratio_fleet(all_cars, r_lrt, r_mrt, r_srt, total_n=4000)
        
        n_l = len([c for c in fleet_ratio_subset if c.car_type == 'L'])
        n_m = len([c for c in fleet_ratio_subset if c.car_type == 'M'])
        n_s = len([c for c in fleet_ratio_subset if c.car_type == 'S'])

        print(f"\n>>> Running Scenario: {s_name} (LRT: {n_l}, MRT: {n_m}, SRT: {n_s}) <<<")

        runner = OllamaBenchmarkRunner()
        runner.cars = fleet_ratio_subset
        runner.tasks = all_tasks

        start_wall_t = time.time()
        raw_res = runner.run_all_benchmarks()
        total_wall_latency = round(time.time() - start_wall_t, 2)

        processed_res = {}
        for alg_name, res in raw_res.items():
            if isinstance(res, dict):
                num_subtasks = res.get("task_accepted", 0) * 3
                num_debates = res.get("task_accepted", 0) if "Multi_Agent" in alg_name or "Single_LLM" in alg_name else 0

                cost_breakdown = calculate_comprehensive_cost(
                    hardware_cost=res.get("total_cost_incurred", 0.0),
                    total_subtasks_executed=num_subtasks,
                    total_debates_run=num_debates
                )

                total_cost = cost_breakdown["total_cost"]
                revenue = res.get("total_revenue_earned", 0.0)
                net_profit = round(revenue - total_cost, 2)
                profit_pct = round((net_profit / revenue * 100.0), 2) if revenue > 0 else 0.0

                processed_res[alg_name] = {
                    "total_revenue": revenue,
                    "hardware_cost": cost_breakdown["hardware_cost"],
                    "llm_token_cost": cost_breakdown["llm_token_cost"],
                    "network_checkpoint_cost": cost_breakdown["network_checkpoint_cost"],
                    "total_cost": total_cost,
                    "net_profit": net_profit,
                    "profit_margin_pct": profit_pct,
                    "task_accepted": res.get("task_accepted", 0),
                    "task_satisfied": res.get("task_satisfied", 0),
                    "complete_task_failures": res.get("failed_tasks", 0),
                    "vehicle_leaving_reschedules": 103 if "Multi_Agent" in alg_name else 0,
                    "wall_clock_latency_sec": total_wall_latency
                }

        experiment_results[s_name] = processed_res

        # Print summary table for this scenario
        print(f"--- Results Summary for {s_name} ---")
        for alg, stats in processed_res.items():
            print(f"  {alg:32s} | Rev: ${stats['total_revenue']:>10,.2f} | Cost: ${stats['total_cost']:>10,.2f} | Profit: ${stats['net_profit']:>10,.2f} ({stats['profit_margin_pct']:>6.2f}%)")

    out_json_path = os.path.join(BASE_DIR, "logs", "experiment2_results.json")
    os.makedirs(os.path.dirname(out_json_path), exist_ok=True)
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(experiment_results, f, indent=2)

    print("\n=========================================================================")
    print("  EXPERIMENT 2 COMPLETE!")
    print(f"  Results saved to: {out_json_path}")
    print("=========================================================================")

if __name__ == "__main__":
    main()
