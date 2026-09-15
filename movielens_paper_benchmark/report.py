"""Dependency-free report generation for MovieLens benchmark summaries.

The report preserves the standard deviations stored by the benchmark runner.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple, Union


PathLike = Union[str, Path]
ReportPaths = Dict[str, Path]

_RESOURCE_KEYS = ("training_seconds", "inference_seconds")
_MEMORY_KEYS = ("cuda_peak_allocated_mib",)
_PROTOCOL_DIAGNOSTIC_KEYS = (
    "positive_catalog_items",
    "validation_target_positive_support_rate",
    "test_target_positive_support_rate",
)


def _load_summary(summary_or_path: Union[Mapping[str, Any], PathLike]) -> Dict[str, Any]:
    if isinstance(summary_or_path, Mapping):
        return dict(summary_or_path)
    path = Path(summary_or_path)
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError("benchmark summary must be a JSON object")
    return value


def _metric_sort_key(name: str) -> Tuple[int, int, str]:
    base, separator, cutoff = name.partition("@")
    family = {"hit": 0, "ndcg": 1, "recall": 2}.get(base.lower(), 3)
    numeric_cutoff = int(cutoff) if separator and cutoff.isdigit() else 10**9
    return family, numeric_cutoff, name


def _proportion_sort_key(label: str) -> Tuple[int, float, str]:
    try:
        value = float(label.rstrip("%"))
        if label.endswith("%"):
            value /= 100.0
        return 0, -value, label
    except (TypeError, ValueError, AttributeError):
        return 1, 0.0, str(label)


def _display_proportion(label: str) -> str:
    text = str(label)
    if text.endswith("%"):
        return text
    try:
        value = float(text)
    except ValueError:
        return text
    if 0.0 <= value <= 1.0:
        percentage = value * 100.0
        if percentage.is_integer():
            return f"{int(percentage)}%"
        return f"{percentage:g}%"
    return text


def _as_stats(value: Any, context: str) -> Tuple[float, float, int]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must contain mean, std and n")
    missing = {"mean", "std", "n"}.difference(value)
    if missing:
        raise ValueError(f"{context} is missing fields: {sorted(missing)}")
    mean = float(value["mean"])
    std = float(value["std"])
    n = int(value["n"])
    if n <= 0:
        raise ValueError(f"{context}.n must be positive")
    if std < 0.0 or not math.isfinite(mean) or not math.isfinite(std):
        raise ValueError(f"{context} contains invalid statistics")
    return mean, std, n


def _models_for_dataset(dataset: Mapping[str, Any]) -> List[str]:
    result: List[str] = []
    for proportion in dataset.values():
        if not isinstance(proportion, Mapping):
            continue
        for model in proportion:
            if model not in result:
                result.append(str(model))
    return result


def _metrics_for_dataset(dataset: Mapping[str, Any]) -> List[str]:
    names = set()
    for proportion in dataset.values():
        if not isinstance(proportion, Mapping):
            continue
        for record in proportion.values():
            if isinstance(record, Mapping) and isinstance(record.get("metrics"), Mapping):
                names.update(str(name) for name in record["metrics"])
    return sorted(names, key=_metric_sort_key)


def _resource_values(record: Mapping[str, Any]) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key in _RESOURCE_KEYS + _MEMORY_KEYS:
        if key in record:
            result[key] = record[key]
    memory = record.get("memory")
    if isinstance(memory, Mapping):
        for key, value in memory.items():
            result[str(key)] = value
    target_support = record.get("target_support")
    if isinstance(target_support, Mapping):
        for key, value in target_support.items():
            result[str(key)] = value
    return result


def _resources_for_dataset(dataset: Mapping[str, Any]) -> List[str]:
    names = set()
    for proportion in dataset.values():
        if not isinstance(proportion, Mapping):
            continue
        for record in proportion.values():
            if isinstance(record, Mapping):
                names.update(_resource_values(record))
    preferred = list(
        _RESOURCE_KEYS + _MEMORY_KEYS + _PROTOCOL_DIAGNOSTIC_KEYS
    )
    return sorted(names, key=lambda name: (preferred.index(name) if name in preferred else 99, name))


def _unit_for_resource(name: str) -> str:
    if name.endswith("_seconds"):
        return "s"
    if name.endswith("_mib") or "_mib_" in name:
        return "MiB"
    if name.endswith("_rate"):
        return "fraction"
    if name.endswith("_items"):
        return "count"
    return ""


def _format_stats(value: Any, decimals: int, context: str) -> str:
    mean, std, n = _as_stats(value, context)
    return f"{mean:.{decimals}f} ± {std:.{decimals}f} (n={n})"


def _latex_escape(value: Any) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in text)


def _markdown_escape(value: Any) -> str:
    return str(value).replace("|", r"\|").replace("\n", " ")


def _write_long_metrics(summary: Mapping[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("dataset", "proportion", "metric", "model", "mean", "std", "n"))
        for dataset_name, dataset in summary.items():
            if not isinstance(dataset, Mapping):
                continue
            for proportion_name in sorted(dataset, key=_proportion_sort_key):
                proportion = dataset[proportion_name]
                if not isinstance(proportion, Mapping):
                    continue
                for model_name, record in proportion.items():
                    if not isinstance(record, Mapping) or not isinstance(record.get("metrics"), Mapping):
                        continue
                    for metric_name in sorted(record["metrics"], key=_metric_sort_key):
                        mean, std, n = _as_stats(
                            record["metrics"][metric_name],
                            f"{dataset_name}/{proportion_name}/{model_name}/{metric_name}",
                        )
                        writer.writerow(
                            (
                                dataset_name,
                                proportion_name,
                                metric_name,
                                model_name,
                                repr(mean),
                                repr(std),
                                n,
                            )
                        )


def _write_long_resources(summary: Mapping[str, Any], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ("dataset", "proportion", "resource", "model", "mean", "std", "n", "unit")
        )
        for dataset_name, dataset in summary.items():
            if not isinstance(dataset, Mapping):
                continue
            for proportion_name in sorted(dataset, key=_proportion_sort_key):
                proportion = dataset[proportion_name]
                if not isinstance(proportion, Mapping):
                    continue
                for model_name, record in proportion.items():
                    if not isinstance(record, Mapping):
                        continue
                    resources = _resource_values(record)
                    for resource_name in _resources_for_dataset({str(proportion_name): proportion}):
                        if resource_name not in resources:
                            continue
                        mean, std, n = _as_stats(
                            resources[resource_name],
                            f"{dataset_name}/{proportion_name}/{model_name}/{resource_name}",
                        )
                        writer.writerow(
                            (
                                dataset_name,
                                proportion_name,
                                resource_name,
                                model_name,
                                repr(mean),
                                repr(std),
                                n,
                                _unit_for_resource(resource_name),
                            )
                        )


def _selected_parameters(summary: Mapping[str, Any]) -> Dict[str, Any]:
    selected: Dict[str, Any] = {}
    for dataset_name, dataset in summary.items():
        if not isinstance(dataset, Mapping):
            continue
        selected[str(dataset_name)] = {}
        for proportion_name in sorted(dataset, key=_proportion_sort_key):
            proportion = dataset[proportion_name]
            if not isinstance(proportion, Mapping):
                continue
            selected[str(dataset_name)][str(proportion_name)] = {}
            for model_name, record in proportion.items():
                if isinstance(record, Mapping) and "parameters" in record:
                    selected[str(dataset_name)][str(proportion_name)][str(model_name)] = record[
                        "parameters"
                    ]
    return selected


def _markdown_table(headers: Sequence[str], rows: Iterable[Sequence[str]]) -> List[str]:
    lines = [
        "| " + " | ".join(_markdown_escape(value) for value in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(_markdown_escape(value) for value in row) + " |" for row in rows)
    return lines


def _write_markdown(summary: Mapping[str, Any], selected: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# MovieLens benchmark report",
        "",
        "Values are means across final runs. Deviations are the standard deviations "
        "stored by the benchmark runner; `n` is the number of final runs.",
    ]
    for dataset_name, dataset in summary.items():
        if not isinstance(dataset, Mapping):
            continue
        models = _models_for_dataset(dataset)
        metrics = _metrics_for_dataset(dataset)
        resources = _resources_for_dataset(dataset)
        lines.extend(("", f"## {_markdown_escape(dataset_name)}", "", "### Ranking metrics", ""))
        metric_rows: List[List[str]] = []
        for proportion_name in sorted(dataset, key=_proportion_sort_key):
            proportion = dataset[proportion_name]
            if not isinstance(proportion, Mapping):
                continue
            for metric_name in metrics:
                row = [_display_proportion(str(proportion_name)), metric_name]
                for model_name in models:
                    record = proportion.get(model_name, {})
                    value = record.get("metrics", {}).get(metric_name) if isinstance(record, Mapping) else None
                    row.append(
                        "—"
                        if value is None
                        else _format_stats(
                            value,
                            4,
                            f"{dataset_name}/{proportion_name}/{model_name}/{metric_name}",
                        )
                    )
                metric_rows.append(row)
        lines.extend(_markdown_table(["Training ratings", "Metric", *models], metric_rows))

        lines.extend(
            (
                "",
                "### Training, inference, memory and protocol diagnostics",
                "",
            )
        )
        if resources:
            resource_rows: List[List[str]] = []
            for proportion_name in sorted(dataset, key=_proportion_sort_key):
                proportion = dataset[proportion_name]
                if not isinstance(proportion, Mapping):
                    continue
                for resource_name in resources:
                    row = [
                        _display_proportion(str(proportion_name)),
                        resource_name,
                        _unit_for_resource(resource_name) or "—",
                    ]
                    for model_name in models:
                        record = proportion.get(model_name, {})
                        value = (
                            _resource_values(record).get(resource_name)
                            if isinstance(record, Mapping)
                            else None
                        )
                        row.append(
                            "—"
                            if value is None
                            else _format_stats(
                                value,
                                3,
                                f"{dataset_name}/{proportion_name}/{model_name}/{resource_name}",
                            )
                        )
                    resource_rows.append(row)
            lines.extend(
                _markdown_table(
                    ["Training ratings", "Resource", "Unit", *models], resource_rows
                )
            )
        else:
            lines.append("Timing and memory statistics were not present in the summary.")

        lines.extend(("", "### Selected hyperparameters", ""))
        parameter_rows = []
        dataset_parameters = selected.get(dataset_name, {})
        for proportion_name in sorted(dataset_parameters, key=_proportion_sort_key):
            for model_name, parameters in dataset_parameters[proportion_name].items():
                compact = json.dumps(parameters, ensure_ascii=False, sort_keys=True, separators=(",", ": "))
                parameter_rows.append(
                    [_display_proportion(str(proportion_name)), str(model_name), f"`{compact}`"]
                )
        lines.extend(_markdown_table(["Training ratings", "Model", "Parameters"], parameter_rows))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _latex_table(
    caption: str,
    label: str,
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
) -> List[str]:
    alignment = "ll" + "c" * (len(headers) - 2)
    lines = [
        r"\begin{table*}[htbp]",
        r"\centering",
        r"\small",
        f"\\begin{{tabular}}{{{alignment}}}",
        r"\hline",
        " & ".join(_latex_escape(value) for value in headers) + r" \\",
        r"\hline",
    ]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.extend(
        (
            r"\hline",
            r"\end{tabular}",
            f"\\caption{{{_latex_escape(caption)}}}",
            f"\\label{{{_latex_escape(label)}}}",
            r"\end{table*}",
        )
    )
    return lines


def _latex_stats(value: Any, decimals: int, context: str) -> str:
    mean, std, n = _as_stats(value, context)
    return f"${mean:.{decimals}f} \\pm {std:.{decimals}f}$ $(n={n})$"


def _write_latex(summary: Mapping[str, Any], path: Path) -> None:
    lines = [
        "% Generated by movielens_paper_benchmark.report.",
        "% Deviations are the standard deviations stored by the benchmark runner.",
    ]
    for dataset_name, dataset in summary.items():
        if not isinstance(dataset, Mapping):
            continue
        models = _models_for_dataset(dataset)
        metrics = _metrics_for_dataset(dataset)
        resources = _resources_for_dataset(dataset)
        metric_rows: List[List[str]] = []
        for proportion_name in sorted(dataset, key=_proportion_sort_key):
            proportion = dataset[proportion_name]
            if not isinstance(proportion, Mapping):
                continue
            for metric_name in metrics:
                row = [
                    _latex_escape(_display_proportion(str(proportion_name))),
                    _latex_escape(metric_name),
                ]
                for model_name in models:
                    record = proportion.get(model_name, {})
                    value = record.get("metrics", {}).get(metric_name) if isinstance(record, Mapping) else None
                    row.append(
                        "--"
                        if value is None
                        else _latex_stats(
                            value,
                            4,
                            f"{dataset_name}/{proportion_name}/{model_name}/{metric_name}",
                        )
                    )
                metric_rows.append(row)
        lines.extend(
            _latex_table(
                f"Paper-compatible sampled-100 test metrics on {dataset_name}. Values are mean, standard deviation and number of runs.",
                f"tab:{dataset_name}-metrics",
                ["Training ratings", "Metric", *models],
                metric_rows,
            )
        )
        lines.append("")

        if resources:
            resource_rows: List[List[str]] = []
            for proportion_name in sorted(dataset, key=_proportion_sort_key):
                proportion = dataset[proportion_name]
                if not isinstance(proportion, Mapping):
                    continue
                for resource_name in resources:
                    row = [
                        _latex_escape(_display_proportion(str(proportion_name))),
                        _latex_escape(
                            f"{resource_name} [{_unit_for_resource(resource_name)}]"
                            if _unit_for_resource(resource_name)
                            else resource_name
                        ),
                    ]
                    for model_name in models:
                        record = proportion.get(model_name, {})
                        value = (
                            _resource_values(record).get(resource_name)
                            if isinstance(record, Mapping)
                            else None
                        )
                        row.append(
                            "--"
                            if value is None
                            else _latex_stats(
                                value,
                                3,
                                f"{dataset_name}/{proportion_name}/{model_name}/{resource_name}",
                            )
                        )
                    resource_rows.append(row)
            lines.extend(
                _latex_table(
                    f"Training, inference, memory and protocol diagnostics on {dataset_name}.",
                    f"tab:{dataset_name}-resources",
                    ["Training ratings", "Resource", *models],
                    resource_rows,
                )
            )
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_search_audit(
    records: Sequence[Mapping[str, Any]], output_dir: PathLike
) -> ReportPaths:
    """Write the complete validation-search trail and embedding-level summary."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    configurations_path = output / "search_configurations.csv"
    embeddings_path = output / "embedding_validation_summary.csv"

    with configurations_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "dataset",
                "proportion",
                "model",
                "objective",
                "config_index",
                "embedding_dim",
                "mean_validation_score",
                "n_validation_seeds",
                "validation_seed_values_json",
                "validation_scores_json",
                "parameters_json",
            )
        )
        for record in records:
            parameters = dict(record["parameters"])
            scores = [float(value) for value in record["validation_scores"]]
            seed_values = [
                int(value) for value in record["validation_seed_values"]
            ]
            if len(seed_values) != len(scores):
                raise ValueError("validation seed/score counts differ")
            writer.writerow(
                (
                    record["dataset"],
                    record["proportion"],
                    record["model"],
                    record["objective"],
                    record["config_index"],
                    parameters.get("embedding_dim", "n/a"),
                    repr(float(record["mean_validation_score"])),
                    len(scores),
                    json.dumps(seed_values, separators=(",", ":")),
                    json.dumps(scores, separators=(",", ":")),
                    json.dumps(
                        parameters,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                )
            )

    groups: Dict[
        Tuple[str, float, str, str, str], List[Mapping[str, Any]]
    ] = {}
    for record in records:
        parameters = dict(record["parameters"])
        embedding = str(parameters.get("embedding_dim", "n/a"))
        key = (
            str(record["dataset"]),
            float(record["proportion"]),
            str(record["model"]),
            str(record["objective"]),
            embedding,
        )
        groups.setdefault(key, []).append(record)

    with embeddings_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            (
                "dataset",
                "proportion",
                "model",
                "objective",
                "embedding_dim",
                "n_configurations",
                "mean_of_validation_means",
                "std_of_validation_means",
                "best_validation_score",
                "best_config_index",
                "best_parameters_json",
            )
        )
        for key in sorted(
            groups,
            key=lambda value: (
                value[0],
                -value[1],
                value[2],
                value[3],
                (
                    -1
                    if value[4] == "n/a"
                    else int(value[4])
                ),
            ),
        ):
            group = groups[key]
            scores = [
                float(record["mean_validation_score"]) for record in group
            ]
            mean = sum(scores) / len(scores)
            std = math.sqrt(
                sum((score - mean) ** 2 for score in scores) / len(scores)
            )
            best = max(
                group,
                key=lambda record: (
                    float(record["mean_validation_score"]),
                    -int(record["config_index"]),
                ),
            )
            writer.writerow(
                (
                    key[0],
                    key[1],
                    key[2],
                    key[3],
                    key[4],
                    len(group),
                    repr(mean),
                    repr(std),
                    repr(float(best["mean_validation_score"])),
                    int(best["config_index"]),
                    json.dumps(
                        dict(best["parameters"]),
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                )
            )

    return {
        "search_configurations_csv": configurations_path,
        "embedding_validation_summary_csv": embeddings_path,
    }


def write_reports(
    summary_or_path: Union[Mapping[str, Any], PathLike], output_dir: PathLike
) -> ReportPaths:
    """Write benchmark reports and return their paths.

    ``summary_or_path`` accepts either ``BenchmarkRunner.summarize``'s mapping
    or the resulting ``summary.json`` path. Existing report files in
    ``output_dir`` are replaced.
    """
    summary = _load_summary(summary_or_path)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "metrics_csv": output / "metrics_long.csv",
        "resources_csv": output / "resources_long.csv",
        "hyperparameters_json": output / "selected_hyperparameters.json",
        "markdown": output / "report.md",
        "latex": output / "report.tex",
    }
    selected = _selected_parameters(summary)
    _write_long_metrics(summary, paths["metrics_csv"])
    _write_long_resources(summary, paths["resources_csv"])
    paths["hyperparameters_json"].write_text(
        json.dumps(selected, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_markdown(summary, selected, paths["markdown"])
    _write_latex(summary, paths["latex"])
    return paths


__all__ = ["write_reports", "write_search_audit", "ReportPaths"]
