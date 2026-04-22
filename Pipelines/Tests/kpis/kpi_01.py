from typing import Any

from Pipelines.run_tests import KPIScore, PipelineResult
from Pipelines.Tests.kpis.common import exact_match, partial_correctness


def evaluate_kpi_01(result: PipelineResult, gold_standard: list[dict[str, Any]]) -> KPIScore:
    parsed = result.action_sequence
    match = 1.0 if exact_match(parsed, gold_standard) else 0.0
    partial_scores = partial_correctness(parsed, gold_standard)

    return KPIScore(
        kpi_id="KPI-01",
        score=match,
        details={
            "exact_match": match,
            "partial_correctness": partial_scores,
        },
    )