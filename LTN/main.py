"""Run the MindReader MF and LTN experiments."""
import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=["mf", "ltn", "ltn-genres"],
                        default=["mf", "ltn", "ltn-genres"])
    parser.add_argument("--proportions", nargs="+", type=float, default=[1.0, 0.8, 0.6, 0.4, 0.2])
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data", type=Path, default=Path("dataset/mr_ntp_all_entities"))
    parser.add_argument("--output", type=Path, default=Path("results/mindreader.json"))
    parser.add_argument("--model-dir", type=Path, default=Path("saved_models"))
    args = parser.parse_args()
    if args.runs < 1 or any(not 0 < p <= 1 for p in args.proportions):
        parser.error("runs must be positive and proportions must be in (0, 1]")

    from ltnrec import evaluation
    from ltnrec.data import MindReaderDataset
    from ltnrec.utils import generate_report_dict, generate_report_table

    evaluation.MODEL_DIR = str(args.model_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    evaluation.experiment(
        models=args.models, data=MindReaderDataset(str(args.data)),
        proportions=args.proportions, test_metrics=["hit@10", "ndcg@10"],
        report_save_path=str(args.output), starting_seed=args.seed,
        n_runs=args.runs, parallelize=False,
    )
    summary = args.output.with_name(args.output.stem + "_summary.json")
    generate_report_dict(str(args.output), str(summary))
    print(generate_report_table(str(summary)))


if __name__ == "__main__":
    main()
