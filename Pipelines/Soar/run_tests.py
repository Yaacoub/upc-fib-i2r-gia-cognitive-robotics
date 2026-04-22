import os
import re
import subprocess
import time
from pathlib import Path
from typing import Optional

from Pipelines.load_dataset import Command, GroundTruth
from Pipelines.parse_language import ActionSequence, CommandID, ParseTime
from Pipelines.run_tests import Evaluation, ExecutionMetrics, PipelineResult, Script, TokenUsage, extract_final_state, map_error_type, run_tests as original_run_tests


def _execute_script(command_id: CommandID, command: Command, ground_truth: GroundTruth, action_sequence: ActionSequence, script: Script, token_usage: TokenUsage, parse_time: ParseTime) -> PipelineResult:
        base_path = Path(__file__).resolve().parent
        cli_path = base_path / "SoarSuite_9.6.4-Multiplatform" / "SoarCLI.sh"
        run_path = base_path / f"run_{command_id}.soar"
        with open(run_path, "w") as f:
            f.write(script)

        soar_start = time.perf_counter()
        process = subprocess.run(
            [str(cli_path)],
            input=f"source {run_path.name}\n",
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=base_path,
            timeout=60,
        )
        soar_time = time.perf_counter() - soar_start

        os.remove(run_path)

        error_type = None
        success = False
        trace_output = _filter_soar_output(process.stdout)

        if "[STUCK]" in process.stdout:
            error_type = map_error_type("stuck")
        elif "[FAILURE]" in process.stdout:
            reason = _extract_failure_reason(trace_output)
            error_type = map_error_type(reason) if reason else "world-knowledge-error"
        elif process.returncode != 0:
            error_type = "linguistic-parsing-error"
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
                llm_parse_time=parse_time,
                symbolic_reasoning_time=soar_time,
                token_usage=TokenUsage(
                    prompt=token_usage.prompt,
                    completion=token_usage.completion,
                    total=token_usage.total,
                ),
            ),
            final_state=extract_final_state(trace_output),
            trace_output=trace_output,
        )


def _extract_failure_reason(trace_output: str) -> Optional[str]:
    """Parse the constraint reason out of a [FAILURE] trace line."""
    match = re.search(r"\[FAILURE\] Command rejected by constraint: (\S+?)\.?$",
                      trace_output, re.MULTILINE)
    return match.group(1) if match else None


def _filter_soar_output(raw_output: str) -> str:
        lines = raw_output.split("\n")
        filtered_lines: list[str] = []

        skip_keywords = [
            "Soar Command Line Interface", "Launching the Soar Cognitive Architecture",
            "...created Soar kernel", "...created agent", "Soar CLI in single agent mode",
            "soar %", "System halted", "Interrupt received", "This Agent halted", "****",
            "Trace level 1 enabled", "For a full list of trace options", "An agent halted during the run",
            "Total: ", "productions sourced", "production excised", "-->", "Sourcing ",
            "--- STEP-", "--- CMD-", "Run stopped"
        ]

        for line in lines:
            stripped = line.strip()
            if not stripped or stripped in ("|", "*"):
                continue
            if any(keyword in line for keyword in skip_keywords):
                continue
            if line.startswith("     ") and ":" in line and "O:" not in line and "==>S:" not in line:
                continue
            filtered_lines.append(line)

        return "\n".join(filtered_lines)


def run_tests(run: int, dataset: list[tuple[Command, GroundTruth]], actions: dict[CommandID, ActionSequence], scripts: dict[CommandID, Script], token_usages: dict[CommandID, TokenUsage], parse_times: dict[CommandID, ParseTime]) -> list[Evaluation]:
    return original_run_tests(run, dataset, actions, scripts, token_usages, parse_times, _execute_script)
