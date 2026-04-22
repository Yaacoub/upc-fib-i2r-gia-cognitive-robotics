import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, cast, Optional

from Pipelines.load_dataset import Command, GroundTruth
from Pipelines.parse_language import ActionSequence, CommandID, ParseTime, TokenUsage


Script = str
State = dict[str, Optional[str]]


_ERROR_TAXONOMY: dict[str, str] = {
    # linguistic-parsing-error: LLM produced an unrecognised action verb
    "unknown-action":           "linguistic-parsing-error",
    # semantic-error: action verb recognised but required semantic slot absent
    "missing-target-key":       "semantic-error",
    "missing-destination-key":  "semantic-error",
    # planning-error: correct world facts, wrong sequencing / no progress
    "not-holding":              "planning-error",
    "stuck":                    "planning-error",
    # world-knowledge-error: entity or location absent / inaccessible in world
    "unresolved-target":        "world-knowledge-error",
    "unresolved-destination":   "world-knowledge-error",
    "unknown-destination":      "world-knowledge-error",
    "non-manipulable":          "world-knowledge-error",
    "capacity-limit":           "world-knowledge-error",
    "out-of-bounds":            "world-knowledge-error",
    "unresolved-agent":         "world-knowledge-error",
}


def map_error_type(internal_reason: Optional[str]) -> Optional[str]:
    if internal_reason is None:
        return None
    return _ERROR_TAXONOMY.get(internal_reason, internal_reason)
    

@dataclass
class KPIScore:
    kpi_id: str
    score: float
    details: dict[str, Any] = field(default_factory=lambda: cast(dict[str, Any], {}))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kpi_id": self.kpi_id,
            "score": self.score,
            "details": self.details,
        }
    

@dataclass
class ExecutionMetrics:
    llm_parse_time: Optional[float] = None
    symbolic_reasoning_time: Optional[float] = None
    token_usage: TokenUsage = field(default_factory=TokenUsage)


@dataclass
class PipelineResult:
    command_id: str
    command: str
    ground_truth: GroundTruth
    action_sequence: list[dict[str, Any]]
    success: bool
    error_type: Optional[str] = None
    metrics: ExecutionMetrics = field(default_factory=ExecutionMetrics)
    final_state: dict[str, Any] = field(default_factory=lambda: cast(dict[str, Any], {})) 
    trace_output: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "command_id": self.command_id,
            "command": self.command,
            "ground_truth": self.ground_truth,
            "action_sequence": self.action_sequence,
            "success": self.success,
            "error_type": self.error_type,
            "metrics": {
                "llm_parse_time": self.metrics.llm_parse_time,
                "symbolic_reasoning_time": self.metrics.symbolic_reasoning_time,
                "token_usage": self.metrics.token_usage.to_dict(),
            },
            "final_state": self.final_state,
            "trace_output": self.trace_output,
        }
    

from Pipelines.Tests.kpis.kpi_01 import evaluate_kpi_01
from Pipelines.Tests.kpis.kpi_02 import evaluate_kpi_02
from Pipelines.Tests.kpis.kpi_03 import evaluate_kpi_03
from Pipelines.Tests.kpis.kpi_04 import evaluate_kpi_04
from Pipelines.Tests.kpis.kpi_05 import evaluate_kpi_05


Evaluation = tuple[PipelineResult, list[KPIScore]]


def extract_final_state(trace_output: str) -> State:
        state: State = {"agent_location": None, "holding": None}

        for line in trace_output.splitlines():
            move_match = re.search(r"\[ACTION\] Agent moved from \S+ to (\S+)", line)
            get_match = re.search(r"\[ACTION\] Agent got (\S+) at \S+", line)
            set_match = re.search(r"\[ACTION\] Agent set (\S+) at (\S+)", line)

            if move_match:
                state["agent_location"] = move_match.group(1)
            elif get_match:
                state["holding"] = get_match.group(1)
            elif set_match:
                state["holding"] = None
                state["last_set_location"] = set_match.group(2)

        return state


def save_tests(evaluations: list[Evaluation], pipeline_name: str):
        pipeline_dir_path = Path(__file__).resolve().parent / "Tests" / "outputs" / pipeline_name
        os.makedirs(pipeline_dir_path, exist_ok=True)
        
        for eval in evaluations:
            out_file = os.path.join(pipeline_dir_path, f"{eval[0].command_id}.json")
            with open(out_file, "w") as f:
                f.write(json.dumps([eval[0].to_dict(), [kpi.to_dict() for kpi in eval[1]]], indent=2))


def summarize_results(evaluations: list[Evaluation], pipeline_name: str):
        kpi_01: list[float] = []
        kpi_02: list[float] = []
        kpi_03: list[float] = []
        kpi_04: list[float] = []
        kpi_05: list[float] = []
        
        for eval in evaluations:
            kpi_01.append(eval[1][0].score)
            kpi_02.append(eval[1][1].score)
            kpi_03.append(eval[1][2].score)
            kpi_04.append(eval[1][3].score)
            kpi_05.append(eval[1][4].score)
        
        def safe_avg(lst: list[float]) -> float:
            return sum(lst) / len(lst) if lst else 0.0
        
        print("\n=== SUMMARY ===")
        print(f"Pipeline: {pipeline_name}")
        print(f"Total Commands: {len(evaluations)}")
        print(f"KPI-01: {safe_avg(kpi_01):.2%}")
        print(f"KPI-02: {safe_avg(kpi_02):.2%}")
        print(f"KPI-03: {safe_avg(kpi_03):.3f}s avg")
        print(f"KPI-04: {safe_avg(kpi_04):.0f} tokens avg")
        print(f"KPI-05: {sum(kpi_05)} total")


def run_tests(run: int, dataset: list[tuple[Command, GroundTruth]], actions: dict[CommandID, ActionSequence], scripts: dict[CommandID, Script], token_usages: dict[CommandID, TokenUsage], parse_times: dict[CommandID, ParseTime], execution_function: Callable[..., PipelineResult]) -> list[Evaluation]:
    output: list[Evaluation] = []

    for idx, (command, _) in enumerate(dataset, 1):
        base_command_id = f"CMD-{idx:02d}"
        command_id = f"{base_command_id}-R{run:02d}"

        ground_truth = dataset[idx - 1][1]
        action = actions.get(command_id, [])
        script = scripts.get(command_id, "")
        token_usage = token_usages.get(command_id, TokenUsage())
        parse_time = parse_times.get(command_id, None)
        result = execution_function(command_id, command, ground_truth, action, script, token_usage, parse_time)

        kpi_01 = evaluate_kpi_01(result, ground_truth)
        kpi_02 = evaluate_kpi_02(result)
        kpi_03 = evaluate_kpi_03(result)
        kpi_04 = evaluate_kpi_04(result)
        kpi_05 = evaluate_kpi_05(result)

        kpi_scores = [
            kpi_01,
            kpi_02,
            kpi_03,
            kpi_04,
            kpi_05,
        ]

        output.append((result, kpi_scores))
    
    return output