import sys

from Pipelines.load_dataset import load_dataset
from Pipelines.parse_language import parse_language
from Pipelines.ACTR.run_tests import run_tests
from Pipelines.ACTR.translate_rdf_to_actr import translate_rdf_to_actr
from Pipelines.run_tests import save_tests, summarize_results


def main():
    num_commands        = None  # Number of commands processed for testing
    runs                = 3     # Number of runs per command for variability analysis
    rewrite_rdf         = False # Whether to rewrite the RDF translation or reuse existing translation
    rewrite_language    = False # Whether to rewrite the language parsing or reuse cached parses

    for run in range(1, runs+1):

        print(f"=== RUN {run}/{runs} ===")

        # Load dataset
        # =========================
        print("Loading dataset...")
        dataset = load_dataset(max=num_commands)

        # Translate RDF to ACT-R
        # =========================
        print(f"Translating RDF to ACT-R...")
        translate_rdf_to_actr(rewrite=rewrite_rdf)

        # Translate human language to JSON
        # =========================
        print("Translating language to JSON...")
        actions, token_usages, parse_times = parse_language(run=run, dataset=dataset, rewrite=rewrite_language)

        # Run tests
        # =========================
        print("Running tests...")
        results = run_tests(run=run, dataset=dataset, actions=actions, scripts={}, token_usages=token_usages, parse_times=parse_times)
        save_tests(results, "ACTR")

        # Summarize results
        # =========================
        print("Summarizing results...")
        summarize_results(results, "ACTR")

        print("")

    return 0


if __name__ == "__main__":
    sys.exit(main())
