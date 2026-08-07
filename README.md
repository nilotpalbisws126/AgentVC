# AgentVC: Agentic AI Orchestration for Reliable Task Offloading in Vehicular Clouds

Reference implementation and experiment harness for the paper **"AgentVC: Agentic AI
Orchestration for Reliable Task Offloading in Vehicular Clouds."**

AgentVC is an LLM-driven, closed-loop **multi-agent scheduler** for static Vehicular
Clouds (VCs). It orchestrates dynamic subtask partitioning, mixed-mode vehicle (VU)
allocation, adaptive redundancy decay, and reactive fault handling in order to maximize
cloud-provider net profit while honoring strict end-to-end task-completion reliability
(SLA) constraints under stochastic vehicle churn.

This repository contains a deterministic discrete-event simulator, the four collaborating
LLM agents, four comparative baselines, and three self-contained experiment drivers used
to produce the paper's evaluation.

---

## Table of Contents

1. [What the code does](#what-the-code-does)
2. [Repository layout](#repository-layout)
3. [The five scheduling methodologies](#the-five-scheduling-methodologies)
4. [Simulation model](#simulation-model)
5. [The three experiments](#the-three-experiments)
6. [Installation](#installation)
7. [Running the experiments](#running-the-experiments)
8. [Input trace formats](#input-trace-formats)
9. [Outputs](#outputs)
10. [Cost & pricing constants](#cost--pricing-constants)
11. [Notes and caveats](#notes-and-caveats)

---

## What the code does

Each experiment loads real/synthetic vehicular parking traces and a set of long-running
compute tasks, then runs **five scheduling policies** through the same deterministic
event-driven simulator. For every policy the harness records revenue, a decomposed cost
breakdown (VU rental + LLM tokens + network checkpointing), net profit, profit margin,
SLA outcomes, and wall-clock latency, and writes a JSON results file.

The proposed policy replaces the offline lookup tables of prior heuristics with a live
**multi-agent debate**: a Task-Splitting Agent and a VU-Allocation Agent negotiate a
plan, a Judge Agent renders a binding JSON allocation decision, and an Uncertainty Agent
reacts to premature vehicle departures during execution. All LLM inference runs locally
through **Ollama** (default model `llama3.1`).

The three experiments share **one identical codebase** (`agents/`, `simulator_python/`,
`Trace_data/`); they differ only in their top-level driver script, which sweeps a
different variable (fleet size, residency mix, or LLM backend).

---

## Repository layout

Each of the three experiment folders is self-contained and has the same structure:

```
vlsid_experimentN_.../
├── run_experimentN.py            # Experiment driver (sweep logic + cost post-processing)
├── run_all_benchmarks.py         # (exp3) Runs all 5 policies once on default traces
├── run_proposed_only.py          # (exp3) Runs only the proposed multi-agent policy
├── library.txt                   # Dependencies + Ubuntu/Conda/Ollama install guide
├── agents/                       # LLM agents and support wrappers
│   ├── ollama_client.py          # Ollama HTTP client (offline fallback aware)
│   ├── debate_engine.py          # Task-Splitting + VU-Allocation agents + debate loop
│   ├── judge_agent.py            # Judge Agent (binding JSON allocation decision)
│   ├── uncertainty_agent.py      # Reactive fault handler (VU departure, 15-min VM MTTR)
│   ├── single_llm_agent.py       # Monolithic single-prompt LLM baseline (B4)
│   ├── sanity_wrapper.py         # Schema/feasibility validation + retry + static fallback
│   └── logging_config.py         # Structured file/console logging
├── simulator_python/             # Deterministic execution environment
│   ├── engine.py                 # Discrete-event engine, TaskRuntime/SubtaskRuntime
│   ├── ttd_queue.py              # Dynamic Time-To-Departure aging queue (LRT/MRT/SRT)
│   ├── trace_loader.py           # Car/task trace parsers
│   └── runner_ollama.py          # 5-way benchmark runner (all policy loops live here)
├── Trace_data/                   # Vehicle + task traces (see formats below)
└── logs/                         # Results JSON, debate transcripts, agent execution log
```

The heart of every experiment is `simulator_python/runner_ollama.py`, which implements the
main time-stepped loop for each of the five policies.

---

## The five scheduling methodologies

All five are executed by `OllamaBenchmarkRunner.run_all_benchmarks()`.

| ID | Name | Description |
|----|------|-------------|
| **B1** | Static Redundant (Florin et al.) | Full `Tn` task redundancy (fixed `n=3`), no checkpointing/subtasking. A replacement is not checkpoint-populated, so a task fails when all its assigned VUs depart. |
| **B2** | Static Checkpointing (Ghazizadeh et al.) | Two equal-length static subtasks (50%/50% for tasks > 300 min), no spatial redundancy (`n=1`). On departure it attempts a single non-checkpointed replacement. |
| **B3** | Static MT99R SOTA (Sarkar et al. 2025) | Fixed subtasks with a recruiter VU (`VU_R`) and category-preference allocation (`n=3`); recruits replacement VUs on recruiter departure. The prior state-of-the-art. |
| **B4** | Single LLM Agent | Monolithic single-prompt LLM that emits the full allocation JSON in one shot (no debate, no judge). |
| **Proposed** | Agentic Multi-Agent Framework | Closed-loop debate (Task-Splitting ↔ VU-Allocation) → Judge Agent binding decision → reactive Uncertainty Agent, with dynamic variable subtasking, mixed-mode VU allocation, adaptive redundancy decay, and a 15-min VM-provisioning (MTTR) penalty on replacement. |

Each policy's loop enforces the same physical accounting: VU rental cost is charged at
`$5 / min` per allocated VU, revenue is earned only if a task finishes on or before its
deadline (otherwise zero), and departed VUs are removed from the idle/allocated pools.

---

## Simulation model

**Engine (`engine.py`).** A `SimulationEngine` advances time in fixed 10-minute steps
from `t=0` to the last vehicle departure. At each step it processes vehicle arrivals,
vehicle departures, and the active policy's admission/execution logic. Tasks carry
`{arrival_time, execution_time, deadline, price}`; laxity is `deadline − t − remaining_exec`.

**Dynamic TTD aging queue (`ttd_queue.py`).** Each vehicle's residency **class is
recomputed live** from its *time-to-departure* rather than fixed at arrival:

- `L` (LRT) — time-to-departure > 360 min
- `M` (MRT) — 180 ≤ time-to-departure ≤ 360 min
- `S` (SRT) — time-to-departure < 180 min

This means a vehicle "ages" `L → M → S` as it approaches its departure, and the allocation
agents always see an up-to-date idle pool bucketed by class. `get_system_snapshot()`
supplies each agent prompt with idle counts and sample VU IDs per class.

**Agents (`agents/`).**
- **Task-Splitting Agent** (Reliability Advocate): proposes variable-sized subtasks and an
  initial redundancy level `n` from task laxity (front-load protection under tight laxity).
- **VU-Allocation Agent** (Resource Advocate): counters by mixing cheap SRT/MRT VUs into
  short subtasks and reserving scarce LRT anchors for long/critical subtasks.
- **Judge Agent**: reads the debate transcript, reasons in a `<scratchpad>`, and emits a
  binding JSON decision (subtask decomposition, allocated VU IDs, category mix, redundancy,
  recruiter). Enforces adaptive redundancy decay `n₁ ≥ n₂ ≥ … ≥ 1`.
- **Uncertainty Agent**: on mid-subtask VU departure, weighs remaining laxity against a
  15-min VM-provisioning penalty and either recruits a checkpoint-populated replacement or
  aborts the task to stop rental burn.
- **Sanity wrapper**: validates every LLM decision against schema and feasibility rules,
  retries up to 2×, and falls back to a deterministic static rule if the LLM output is
  unusable — so the simulator stays deterministic and never crashes on malformed JSON.

**LLM client (`ollama_client.py`).** Talks to a local Ollama server
(`http://localhost:11434`) with `temperature=0.1`. If the server is unreachable it flips to
an instant offline-fallback mode so the harness can be dry-run without a GPU (the sanity
wrapper then supplies deterministic decisions).

---

## The three experiments

### Experiment 1 — Sensitivity to Vehicular Fleet Size (`vlsid_experiment1_scaling_analysis`)
Sweeps **task count × fleet size** while strictly preserving the native LRT:MRT:SRT ratio
of `car_data.txt` when subsampling.
- Task counts: `{50, 100, 200}`
- Fleet sizes: `{1000, 2000, 4000}` VUs
- Tests scalability and the framework's ability to harvest cheap SRT nodes when scarce LRT
  capacity drops in resource-constrained fleets.
- Driver: `run_experiment1.py` → `logs/experiment1_results.json`

### Experiment 2 — Sensitivity to Fleet Residency Composition (`vlsid_experiment2_residency_ratios`)
Fixes load at **100 tasks / 4,000 VUs** and synthesizes three fleets with different
LRT:MRT:SRT mixes to model different parking venues:
- `Ratio_1_Baseline_Mixed` — 5.1% L / 15.2% M / 79.7% S (native control baseline)
- `Ratio_2_Airport_LongDominant` — 40% L / 40% M / 20% S (overnight/airport)
- `Ratio_3_ShoppingMall_ShortDominant` — 2% L / 8% M / 90% S (high-churn retail)
- Benchmarks how well the VU-Allocation Agent trades cost against reliability under very
  different supply mixes.
- Driver: `run_experiment2.py` → `logs/experiment2_results.json`

### Experiment 3 — Cross-LLM Backend Comparison (`vlsid_experiment3_cross_llm`)
Fixes the workload (**50 tasks, first 2,000 VUs of `car_data.txt`**) and re-runs the full
suite under three different Ollama LLM backends, each with its own token price:
- `llama3.1` — Meta Llama 3.1 8B — `$0.00015 / 1k tokens`
- `deepseek-r1:14b` — DeepSeek-R1 14B (reasoning) — `$0.00055 / 1k tokens`
- `kimina-prover-7b` — Kimina Prover 7B — `$0.00020 / 1k tokens`
- Measures how model choice affects decision quality vs. token cost / profit.
- Driver: `run_experiment3.py` → `logs/experiment3_results.json`
- Extras: `run_all_benchmarks.py` (all 5 policies once on default traces) and
  `run_proposed_only.py` (proposed policy only).

> Each model must be pulled into Ollama first, e.g. `ollama pull deepseek-r1:14b`.

---

## Installation

Target environment: Ubuntu 20.04/22.04/24.04 with an NVIDIA GPU, Python 3.10 (Conda
recommended), and a running Ollama server. Full instructions are in each folder's
`library.txt`. Quick version:

```bash
# 1. Python environment
conda create -n vlsid_env python=3.10 -y
conda activate vlsid_env
pip install --upgrade pip
pip install numpy pandas matplotlib requests pydantic tqdm dataclasses-json

# 2. Ollama + model
curl -fsSL https://ollama.com/install.sh | sh
ollama serve &                 # start the server
ollama pull llama3.1           # default model (also pull deepseek-r1:14b / kimina-prover-7b for exp3)
```

The scheduling code itself only needs the Python standard library plus a reachable Ollama
endpoint; the extra packages in `library.txt` cover optional data handling/plotting.

**Environment variables (optional):**
- `OLLAMA_MODEL` — override the model name
- `OLLAMA_HOST` — override the server URL (default `http://localhost:11434`)

---

## Running the experiments

From inside a given experiment folder:

```bash
cd vlsid_experiment1_scaling_analysis
python run_experiment1.py

cd ../vlsid_experiment2_residency_ratios
python run_experiment2.py

cd ../vlsid_experiment3_cross_llm
python run_experiment3.py
# optional:
python run_all_benchmarks.py
python run_proposed_only.py
```

Each driver prints a per-scenario summary table (revenue / cost / profit / margin) and
saves a JSON file under `logs/`.

---

## Input trace formats

Whitespace-separated, one record per line; `#` lines are ignored.

**Vehicle trace** (`Trace_data/car_data.txt`) — `car_id  arrival  departure  type`
where `type ∈ {0=S(SRT), 1=M(MRT), 2=L(LRT)}` (letters `S/M/L` also accepted). The shipped
`car_data.txt` has 4,000 VUs (3,188 SRT / 608 MRT / 204 LRT).

**Task trace** (`Trace_data/task_data*.txt`) —
`task_id  arrival  execution_time  deadline  price  [packet_id]  [packet_count]`.
All times are in minutes; price follows the paper's non-linear
`p = K₁·e^1.5 + K₂/l²` model. `task_data_100tasks_1000min.txt` (100 tasks) and
`task_data.txt` (1,000 tasks) are the primary task sets.

The `Trace_data/` folder also ships additional real/synthetic parking traces
(`queens_parking_*`, `grand_arcade_*`, `trace_data_1..9`) for further experimentation.

---

## Outputs

Written to each experiment's `logs/` directory:

- `experimentN_results.json` — per-scenario, per-policy metrics: total revenue,
  `hardware_cost` / `llm_token_cost` / `network_checkpoint_cost` / `total_cost`,
  `net_profit`, `profit_margin_pct`, `task_accepted` / `task_satisfied`,
  `complete_task_failures`, `vehicle_leaving_reschedules`, and `wall_clock_latency_sec`.
- `benchmark_results.json` / `full_benchmark_results.json` — raw per-policy engine output.
- `debate_transcripts.jsonl` — one JSON record per task's debate (both agent proposals +
  full transcript).
- `agent_execution.log` — timestamped log of every agent/LLM interaction.

---

## Cost & pricing constants

Defined at the top of each `run_experimentN.py` (and the pricing model in the trace files):

| Constant | Value | Meaning |
|----------|-------|---------|
| `HARDWARE_RENTAL_RATE_PER_MIN` | `$5.00` | VU rental per minute per allocated VU |
| `LLM_TOKEN_COST_PER_1K` | `$0.00015` | LLM inference cost per 1k tokens (per-model in exp3) |
| `NETWORK_CHECKPOINT_COST_PER_GB` | `$0.01` | Network transfer cost per GB |
| `AVG_CHECKPOINT_SIZE_GB` | `2.5` | Assumed data per saved checkpoint |
| `AVG_TOKENS_PER_DEBATE` | `1650` | Assumed tokens per 3-turn debate |
| `MTTR` (VM provisioning) | `15 min` | Replacement-VU image-install penalty |
| `emin` (min subtask) | `60 min` | Minimum subtask length |

---

## Notes and caveats

- **Shared code.** `agents/`, `simulator_python/`, and `Trace_data/` are byte-identical
  across the three experiment folders. Only the driver script and its sweep parameters
  differ. If you patch the engine or an agent, apply the change to all three copies.
- **Determinism.** The environment (arrivals, departures, execution, accounting) is fully
  deterministic; only the LLM reasoning layer is stochastic, and its outputs are bounded by
  the sanity wrapper, which falls back to static rules on invalid/unavailable LLM output.
- **Offline dry-runs.** With no Ollama server reachable, the client enters offline mode and
  the sanity wrapper's deterministic fallback drives the LLM policies — useful for smoke
  testing the pipeline without a GPU (results will not reflect true LLM behavior).
- **Reproducibility.** LLM-backed numbers depend on the exact model weights/quantization
  served by Ollama and its sampling; re-runs can vary slightly. Baselines B1–B3 are
  deterministic.
- **`vehicle_leaving_reschedules`** is recorded as a fixed reporting value for multi-agent
  policies in the current drivers rather than counted live from the engine.
