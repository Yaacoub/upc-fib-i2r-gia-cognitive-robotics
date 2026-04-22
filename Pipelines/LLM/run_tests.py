import json
import re
import time
from pathlib import Path
from typing import Optional

from Pipelines.load_dataset import Command, GroundTruth
from Pipelines.parse_language import ActionSequence, CommandID, ParseTime
from Pipelines.run_tests import Evaluation, ExecutionMetrics, PipelineResult, Script, TokenUsage, extract_final_state, map_error_type, run_tests as original_run_tests
from Pipelines.LLM.plan import plan


def _execute_script(command_id: CommandID, command: Command, ground_truth: GroundTruth, action_sequence: ActionSequence, script: Script, token_usage: TokenUsage, parse_time: ParseTime) -> PipelineResult:
    environment = ""
    environment_path = Path(__file__).resolve().parent / "cognitive-robotics.txt"
    with open(environment_path, "r") as f:
        environment = f.read()
    
    llm_start = time.perf_counter()
    output, plan_token_usage = plan(environment, json.dumps(action_sequence))
    print(f"\r  Planned {command_id}", end="", flush=True)
    llm_time = time.perf_counter() - llm_start

    error_type = None
    success = False

    if "[STUCK]" in output:
        error_type = map_error_type("stuck")
    elif "[FAILURE]" in output:
        reason = _extract_failure_reason(output)
        error_type = map_error_type(reason) if reason else "world-knowledge-error"
    elif "[SUCCESS]" not in output:
        error_type = "linguistic-parsing-error"   # LLM produced malformed trace
    else:
        success = True

    return PipelineResult(
        command_id=command_id,
        command=command,
        ground_truth=ground_truth,
        action_sequence=action_sequence,
        success=success,
        error_type=error_type,
        metrics=ExecutionMetrics(
            llm_parse_time=(parse_time or 0) + (llm_time or 0),
            symbolic_reasoning_time=0,
            token_usage=TokenUsage(
                prompt=(token_usage.prompt or 0) + (plan_token_usage.prompt or 0),
                completion=(token_usage.completion or 0) + (plan_token_usage.completion or 0),
                total=(token_usage.total or 0) + (plan_token_usage.total or 0),
            ),
        ),
        final_state=extract_final_state(output),
        trace_output=output,
    )


def _extract_failure_reason(trace_output: str) -> Optional[str]:
    match = re.search(r"\[FAILURE\] Command rejected by constraint:\s*([a-zA-Z0-9_-]+)", trace_output)
    return match.group(1) if match else None


def run_tests(run: int, dataset: list[tuple[Command, GroundTruth]], actions: dict[CommandID, ActionSequence], scripts: dict[CommandID, Script], token_usages: dict[CommandID, TokenUsage], parse_times: dict[CommandID, ParseTime]) -> list[Evaluation]:
    result = original_run_tests(run, dataset, actions, scripts, token_usages, parse_times, _execute_script)
    print("")
    return result