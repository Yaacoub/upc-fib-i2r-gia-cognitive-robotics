# Cognitive Robotics: Comparative Evaluation of Hybrid LLM-Symbolic Instruction Pipelines

[![Course](https://img.shields.io/badge/Course-Introduction_to_Research-blue.svg)](https://www.fib.upc.edu/en)
[![Institution](https://img.shields.io/badge/Institution-UPC_FIB-red.svg)](https://www.fib.upc.edu/en)

This repository contains the source code, dataset, and evaluation framework for the research paper **"Cognitive Robotics: Comparative Evaluation of Hybrid LLM-Symbolic Instruction Pipelines"** by Peter Yaacoub (Universitat Politècnica de Catalunya).

The project evaluates four instruction-to-action pipelines for cognitive robotics under a shared OWL/RDF world model and a common natural-language benchmark: **Soar**, **ACT-R**, **Ontology Executor**, and a **Pure LLM Baseline**. All pipelines consume the same JSON action schema produced by the same parser, diverging only at execution time to isolate and benchmark their reasoning, routing, and constraint-handling capabilities.

## Key Findings

Following 159 execution runs per pipeline (636 total evaluations), the study supports a practical division of labor: LLMs are highly effective semantic translators, while symbolic/ontological layers provide grounded control and explicit constraint enforcement. 

*   **Accuracy (KPI-01 & 02):** All pipelines reached **100%** action-sequence match to the gold JSON and a **100%** task completion rate (under the study's safe-rejection definition).
*   **Efficiency (KPI-03 & 04):** The hybrid symbolic architectures (Soar, ACT-R, Ontology) averaged **~13.1s** planning latency and **1,286 tokens** per run. The Pure LLM Baseline averaged **~37.2s** and **5,579 tokens** (approx. 65% slower and 77% more token-intensive).
*   **Safety (KPI-05):** Constraint violations (e.g., attempting physically impossible tasks or out-of-bounds navigation) were explicitly caught and safely rejected **45 times** per pipeline, perfectly aligning with the dataset's adversarial intent.

## Project Structure

```text
├── Dataset Creation/
├── Domain Modeling/
├── Paper/
└── Pipelines/
    ├── ACTR/
    ├── LLM/
    ├── Ontology/
    ├── Soar/
    └── Tests/
```

## Dataset & KPIs

The benchmark curates **53 natural language commands** structured across five distinct interaction categories: *Simple*, *Attribute-based*, *Spatial Relations*, *Ambiguous*, and *Multi-step*.

**Gold Standard Format Example** (JSON, parsed by the LLM before symbolic execution):
```json
[{"action": "move", "desired-x": 2, "desired-y": 2}]
[{"action": "get", "target-class": "apple", "target-modifiers": ["green"]}, {"action": "set", "desired-x": 1, "desired-y": 3, "destination-class": "gridlocation"}]
```

The testing harness orchestrates the execution and evaluates each run across **5 Key Performance Indicators**:

| KPI | Name | Metric | Range |
|---|---|---|---|
| **KPI-01** | Interpretation Accuracy | Exact match rate + partial correctness (Action, Object, Relation) | 0.0–1.0 |
| **KPI-02** | Task Success Rate | Pipeline execution success or safe symbolic constraint rejection | 0.0–1.0 |
| **KPI-03** | Planning Latency | Processing time (partitioned into LLM Parse + Symbolic Reasoning) | Seconds |
| **KPI-04** | Model Complexity | Token consumption (Prompt + Completion) | Count |
| **KPI-05** | Constraint Violations | Instances of domain boundaries breached (e.g., limits, grid bounds) | Count |

## Quick Start

Run these steps from the repository root.

### 1. Environment Setup

Create a virtual environment and activate it:
```bash
python3 -m venv .venv
source .venv/bin/activate
```

Upgrade `pip` and install the required Python dependencies:
```bash
python -m pip install --upgrade pip
python -m pip install google-genai python-dotenv rdflib
```

### 2. Configure the LLM Parser
The system uses a foundation model to translate language into the JSON logic. Provide a Gemini API key:
```bash
export GEMINI_API_KEY="your_key_here"
```
*(Alternatively, you can place the same value in a `.env` file at the repository root: `echo "GEMINI_API_KEY=your_key_here" > .env`)*

### 3. Install Soar (Required for Pipeline A)
To run the Soar pipeline, download Soar 9.6.4 from [the official release page](https://github.com/SoarGroup/Soar/releases/tag/releases%2F9.6.4) and extract the `SoarSuite_9.6.4-Multiplatform` folder directly into the `Pipelines/Soar/` directory.

### 4. Run the Pipelines
You can evaluate any of the pipelines independently. Each script automatically loads the 53-command dataset, performs semantic translations, runs **three seeded evaluation passes** (159 runs total), and computes the KPI scores.

```bash
# Run the Soar Cognitive Architecture
python3 -m Pipelines.Soar.soar

# Run the ACT-R Abstraction
python3 -m Pipelines.ACTR.actr

# Run the Ontology Executor
python3 -m Pipelines.Ontology.ontology

# Run the Pure LLM Baseline
python3 -m Pipelines.LLM.llm
```

A summary will be printed to the terminal after each run, and deep-dive JSON metrics (including execution traces and final states) are automatically exported to `Pipelines/Tests/outputs/<Pipeline>/`.