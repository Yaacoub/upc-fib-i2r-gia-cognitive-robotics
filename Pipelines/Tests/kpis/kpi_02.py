from Pipelines.run_tests import KPIScore, PipelineResult


def evaluate_kpi_02(result: PipelineResult) -> KPIScore:
    is_constraint = result.error_type not in [None, "stuck", "execution-error"]
    success = 1.0 if result.success or is_constraint else 0.0

    return KPIScore(
        kpi_id="KPI-02",
        score=success,
        details={
            "task_success": success,
            "pipeline_success": result.success,
            "error_type": result.error_type,
            "final_state": result.final_state,
        },
    )
