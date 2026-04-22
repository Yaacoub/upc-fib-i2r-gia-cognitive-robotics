from Pipelines.run_tests import KPIScore, PipelineResult


def evaluate_kpi_03(result: PipelineResult) -> KPIScore:
    llm = result.metrics.llm_parse_time or 0.0
    symbolic = result.metrics.symbolic_reasoning_time or 0.0

    return KPIScore(
        kpi_id="KPI-03",
        score=llm + symbolic,
        details={
            "llm_parse_time": llm,
            "symbolic_reasoning_time": symbolic,
            "total_time": llm + symbolic,
        },
    )
