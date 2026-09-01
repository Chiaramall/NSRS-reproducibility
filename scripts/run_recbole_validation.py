#!/usr/bin/env python3
import argparse
import json

from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import get_model, get_trainer, init_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config_files", required=True)
    args, _ = parser.parse_known_args()
    config = Config(
        model=args.model,
        dataset=args.dataset,
        config_file_list=args.config_files.split(),
    )
    init_seed(config["seed"], config["reproducibility"])
    dataset = create_dataset(config)
    train_data, valid_data, _ = data_preparation(config, dataset)
    model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
    trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
    best_score, best_result = trainer.fit(
        train_data, valid_data, saved=False, show_progress=config["show_progress"]
    )
    print("VALIDATION_RESULT=" + json.dumps({
        "model": args.model.lower(),
        "dataset": args.dataset,
        "score": float(best_score),
        **{key: float(value) for key, value in best_result.items()},
    }, sort_keys=True))


if __name__ == "__main__":
    main()
