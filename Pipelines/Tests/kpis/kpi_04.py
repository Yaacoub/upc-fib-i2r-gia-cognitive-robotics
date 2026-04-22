from Pipelines.run_tests import KPIScore, PipelineResult


def evaluate_kpi_04(result: PipelineResult) -> KPIScore:
    token_usage = result.metrics.token_usage

    return KPIScore(
        kpi_id="KPI-04",
        score=token_usage.total or 0,
        details={
            "llm_prompt_tokens": token_usage.prompt,
            "llm_completion_tokens": token_usage.completion,
            "llm_total_tokens": token_usage.total,
        },
    )
