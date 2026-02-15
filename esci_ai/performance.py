import json
from dataclasses import asdict, dataclass
from pathlib import Path

from esci_ai.models import MatchClassification, QueryProductExample


@dataclass
class ClassifierPerformance:
    n: int
    accuracy: float
    precision: float
    recall: float
    tp_ids: list[str | int]
    tn_ids: list[str | int]
    fp_ids: list[str | int]
    fn_ids: list[str | int]


def get_classifier_performance(
    predictions: list[QueryProductExample], report_path: str | Path | None = None
) -> ClassifierPerformance | None:

    # filter classified examples
    preds = predictions.copy()
    preds = [
        p
        for p in preds
        if p.query_product_match and p.query_product_match.match_classification
    ]

    n = len(preds)

    # separate classifications: positive = pred exact match, negative = pred not exact match
    positives = [
        p
        for p in preds
        if p.query_product_match.match_classification == MatchClassification.EXACT_MATCH
    ]
    negatives = [
        p
        for p in preds
        if p.query_product_match.match_classification
        == MatchClassification.NOT_EXACT_MATCH
    ]
    true_positives = [e for e in positives if e.example_id not in NEGATIVE_EXAMPLE_IDS]
    true_negatives = [e for e in negatives if e.example_id in NEGATIVE_EXAMPLE_IDS]
    false_positives = [e for e in positives if e.example_id in NEGATIVE_EXAMPLE_IDS]
    false_negatives = [e for e in negatives if e.example_id not in NEGATIVE_EXAMPLE_IDS]

    # note: this is not safe (zero-division possible)
    n = len(preds)
    accuracy = (len(true_positives) + len(true_negatives)) / n
    precision = len(true_positives) / (len(true_positives) + len(false_positives))
    recall = len(true_positives) / (len(true_positives) + len(false_negatives))

    performance = ClassifierPerformance(
        n,
        accuracy,
        precision,
        recall,
        tp_ids=[e.example_id for e in true_positives],
        tn_ids=[e.example_id for e in true_negatives],
        fp_ids=[e.example_id for e in false_positives],
        fn_ids=[e.example_id for e in false_negatives],
    )

    if report_path:
        report = {
            "performance": asdict(performance),
            "false_positives": [e.model_dump() for e in false_positives],
            "false_negatives": [e.model_dump() for e in false_negatives],
        }
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

    return performance


NEGATIVE_EXAMPLE_IDS = [
    # batteries
    142660,  # 60 count, not 100 count
    142666,  # AAA, not AA
    # drills
    660823,  # no mention of gyroscopic
    660827,  # no mention of gyroscopic
    660838,  # charger only, no drill
    # paper
    1163629,  # matte, not glossy
    1163641,  # matte, not glossy
]

AMBIGUOUS_EXAMPLE_IDS = [
    # batteries
    142661,  # ambiguous: title field says 100 count, bullet field says 50 count
    # paper
    1163634,  # ambiguous: title field says Kodak, brand field says Doaaler
]
