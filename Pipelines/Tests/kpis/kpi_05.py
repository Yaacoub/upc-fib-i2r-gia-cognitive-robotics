from Pipelines.run_tests import KPIScore, PipelineResult


def evaluate_kpi_05(result: PipelineResult) -> KPIScore:
    violation = result.error_type not in [None, "stuck", "execution-error"]

    return KPIScore(
        kpi_id="KPI-05",
        score=1 if violation else 0,
        details={
            "violation_count": 1 if violation else 0,
            "violation-type": result.error_type if violation else None,
        },
    )
