import time

from Pipelines.load_dataset import Command, GroundTruth
from Pipelines.Ontology.executor import execute_action_sequence
from Pipelines.Ontology.world_model import OntologyWorld, default_rdf_path
from Pipelines.parse_language import ActionSequence, CommandID, ParseTime
from Pipelines.run_tests import Evaluation, ExecutionMetrics, PipelineResult, Script, TokenUsage, extract_final_state, map_error_type, run_tests as original_run_tests


def _execute_script(command_id: CommandID, command: Command, ground_truth: GroundTruth, action_sequence: ActionSequence, script: Script, token_usage: TokenUsage, parse_time: ParseTime) -> PipelineResult:
    world = OntologyWorld.from_file(default_rdf_path())

    onto_start = time.perf_counter()
    trace_output, failure = execute_action_sequence(world, action_sequence)
    onto_time = time.perf_counter() - onto_start

    error_type = None
    success = False

    if failure is not None:
        error_type = map_error_type(failure)
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
            symbolic_reasoning_time=onto_time,
            token_usage=TokenUsage(
                prompt=token_usage.prompt,
                completion=token_usage.completion,
                total=token_usage.total,
            ),
        ),
        final_state=extract_final_state(trace_output),
        trace_output=trace_output,
    )


def run_tests(run: int, dataset: list[tuple[Command, GroundTruth]], actions: dict[CommandID, ActionSequence], scripts: dict[CommandID, Script], token_usages: dict[CommandID, TokenUsage], parse_times: dict[CommandID, ParseTime]) -> list[Evaluation]:
    return original_run_tests(run, dataset, actions, scripts, token_usages, parse_times, _execute_script)
