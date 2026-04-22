import sys

from Pipelines.load_dataset import load_dataset
from Pipelines.parse_language import parse_language
from Pipelines.Soar.run_tests import run_tests
from Pipelines.Soar.translate_json_to_soar import translate_json_to_soar
from Pipelines.Soar.translate_rdf_to_soar import translate_rdf_to_soar
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

        # Translate RDF to Soar
        # =========================
        print(f"Translating RDF to Soar...")
        translate_rdf_to_soar(rewrite=rewrite_rdf)

        # Translate human language to JSON
        # =========================
        print("Translating language to JSON...")
        actions, token_usages, parse_times = parse_language(run=run, dataset=dataset, rewrite=rewrite_language)

        # Translate JSON to Soar
        # =========================
        print("Translating JSON to Soar...")
        scripts = translate_json_to_soar(actions=actions)

        # Run tests
        # =========================
        print("Running tests...")
        results = run_tests(run=run, dataset=dataset, actions=actions, scripts=scripts, token_usages=token_usages, parse_times=parse_times)
        save_tests(results, "Soar")

        # Summarize results
        # =========================
        print("Summarizing results...")
        summarize_results(results, "Soar")

        print("")

    return 0


if __name__ == "__main__":
    sys.exit(main())
