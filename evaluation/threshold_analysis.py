import json


RESULTS_FILE = "evaluation/results.json"


def load_results():

    with open(
        RESULTS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    return data["results"]


def evaluate_threshold(
    results,
    threshold
):

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    for result in results:

        score = result["top_score"]

        predicted = (
            score >= threshold
        )

        expected = (
            result["expected_answerable"]
        )

        if expected and predicted:
            TP += 1

        elif not expected and not predicted:
            TN += 1

        elif not expected and predicted:
            FP += 1

        elif expected and not predicted:
            FN += 1

    total = len(results)

    accuracy = (
        (TP + TN) / total
        if total
        else 0
    )

    precision = (
        TP / (TP + FP)
        if TP + FP
        else 0
    )

    recall = (
        TP / (TP + FN)
        if TP + FN
        else 0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if precision + recall
        else 0
    )

    false_accept_rate = (
        FP / (FP + TN)
        if FP + TN
        else 0
    )

    false_reject_rate = (
        FN / (FN + TP)
        if FN + TP
        else 0
    )

    return {
        "threshold": threshold,
        "TP": TP,
        "TN": TN,
        "FP": FP,
        "FN": FN,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "FAR": false_accept_rate,
        "FRR": false_reject_rate
    }


def main():

    results = load_results()

    print("=" * 100)
    print("EVIDENCE THRESHOLD ANALYSIS")
    print("=" * 100)

    print()

    thresholds = [
        -12,
        -10,
        -8,
        -6,
        -4,
        -2,
        0,
        1,
        2,
        2.5,
        3,
        3.5,
        4,
        4.5,
        5
    ]

    print(
        f"{'Threshold':>10} "
        f"{'TP':>5} "
        f"{'TN':>5} "
        f"{'FP':>5} "
        f"{'FN':>5} "
        f"{'Precision':>10} "
        f"{'Recall':>10} "
        f"{'F1':>10} "
        f"{'FAR':>10} "
        f"{'FRR':>10}"
    )

    print("-" * 100)

    best = None

    for threshold in thresholds:

        metrics = evaluate_threshold(
            results,
            threshold
        )

        print(
            f"{metrics['threshold']:>10.2f} "
            f"{metrics['TP']:>5} "
            f"{metrics['TN']:>5} "
            f"{metrics['FP']:>5} "
            f"{metrics['FN']:>5} "
            f"{metrics['precision']:>10.3f} "
            f"{metrics['recall']:>10.3f} "
            f"{metrics['f1']:>10.3f} "
            f"{metrics['FAR']:>10.3f} "
            f"{metrics['FRR']:>10.3f}"
        )

        if (
            best is None
            or metrics["f1"] > best["f1"]
        ):
            best = metrics

    print()
    print("=" * 100)
    print("BEST THRESHOLD")
    print("=" * 100)

    for key, value in best.items():

        print(
            f"{key:15}: {value}"
        )


if __name__ == "__main__":
    main()