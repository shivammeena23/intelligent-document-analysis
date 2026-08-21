import json
import time
from pathlib import Path

from app.query_engine import QueryEngine


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

QUESTIONS_FILE = Path(
    "evaluation/questions.json"
)

RESULTS_FILE = Path(
    "evaluation/results.json"
)


# ---------------------------------------------------------
# Load evaluation questions
# ---------------------------------------------------------

def load_questions():

    with open(
        QUESTIONS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ---------------------------------------------------------
# Save results
# ---------------------------------------------------------

def save_results(results):

    metrics = calculate_metrics(results)

    output = {
        "metrics": metrics,
        "results": results
    }

    with open(
        RESULTS_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            output,
            file,
            indent=4,
            ensure_ascii=False
        )


# ---------------------------------------------------------
# Evaluate one question
# ---------------------------------------------------------

def evaluate_question(
    engine,
    question_data
):

    question = question_data["question"]

    expected_answerable = (
        question_data["answerable"]
    )

    start_time = time.perf_counter()

    try:

        result = engine.answer(
            query=question,
            top_k=5
        )

        end_time = time.perf_counter()

        latency = (
            end_time - start_time
        )

        evidence = result.get(
            "evidence",
            {}
        )

        status = evidence.get(
            "status",
            "unknown"
        )

        predicted_answerable = (
            status == "sufficient"
        )

        top_score = evidence.get(
            "top_score"
        )

        relevant_chunks = evidence.get(
            "relevant_chunks",
            0
        )

        return {

            "id": question_data["id"],

            "question": question,

            "expected_answerable":
                expected_answerable,

            "predicted_answerable":
                predicted_answerable,

            "evidence_status":
                status,

            "top_score":
                top_score,

            "relevant_chunks":
                relevant_chunks,

            "latency_seconds":
                round(
                    latency,
                    4
                ),

            "answer":
                result.get(
                    "answer",
                    ""
                ),

            "error":
                None
        }

    except Exception as error:

        end_time = time.perf_counter()

        latency = (
            end_time - start_time
        )

        print(
            "\n⚠ ERROR while processing "
            f"question {question_data['id']}"
        )

        print(
            f"  {type(error).__name__}: {error}"
        )

        return {

            "id": question_data["id"],

            "question": question,

            "expected_answerable":
                expected_answerable,

            "predicted_answerable":
                None,

            "evidence_status":
                "error",

            "top_score":
                None,

            "relevant_chunks":
                0,

            "latency_seconds":
                round(
                    latency,
                    4
                ),

            "answer":
                "",

            "error":
                str(error)
        }


# ---------------------------------------------------------
# Calculate metrics
# ---------------------------------------------------------

def calculate_metrics(results):

    TP = 0
    TN = 0
    FP = 0
    FN = 0

    evaluated_questions = 0

    for result in results:

        predicted = (
            result["predicted_answerable"]
        )

        expected = (
            result["expected_answerable"]
        )

        # ---------------------------------------------
        # Ignore questions that failed because of
        # external errors such as Gemini quota.
        # ---------------------------------------------

        if predicted is None:
            continue

        evaluated_questions += 1

        if expected and predicted:

            TP += 1

        elif not expected and not predicted:

            TN += 1

        elif not expected and predicted:

            FP += 1

        elif expected and not predicted:

            FN += 1

    total = evaluated_questions

    accuracy = (
        (TP + TN) / total
        if total > 0
        else 0
    )

    precision = (
        TP / (TP + FP)
        if (TP + FP) > 0
        else 0
    )

    recall = (
        TP / (TP + FN)
        if (TP + FN) > 0
        else 0
    )

    f1 = (
        2 * precision * recall
        / (precision + recall)
        if (precision + recall) > 0
        else 0
    )

    false_accept_rate = (
        FP / (FP + TN)
        if (FP + TN) > 0
        else 0
    )

    false_reject_rate = (
        FN / (FN + TP)
        if (FN + TP) > 0
        else 0
    )

    successful_results = [
        r
        for r in results
        if r["error"] is None
    ]

    average_latency = (
        sum(
            r["latency_seconds"]
            for r in successful_results
        )
        / len(successful_results)
        if successful_results
        else 0
    )

    failed_questions = (
        len(results)
        - len(successful_results)
    )

    return {

        "total_questions":
            len(results),

        "evaluated_questions":
            evaluated_questions,

        "failed_questions":
            failed_questions,

        "true_positive":
            TP,

        "true_negative":
            TN,

        "false_positive":
            FP,

        "false_negative":
            FN,

        "accuracy":
            round(
                accuracy,
                4
            ),

        "precision":
            round(
                precision,
                4
            ),

        "recall":
            round(
                recall,
                4
            ),

        "f1_score":
            round(
                f1,
                4
            ),

        "false_accept_rate":
            round(
                false_accept_rate,
                4
            ),

        "false_reject_rate":
            round(
                false_reject_rate,
                4
            ),

        "average_latency_seconds":
            round(
                average_latency,
                4
            )
    }


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

def main():

    print("=" * 80)
    print("RAG EVALUATION")
    print("=" * 80)

    questions = load_questions()

    print(
        f"\nLoaded {len(questions)} questions."
    )

    print(
        "\nInitializing Query Engine..."
    )

    engine = QueryEngine()

    results = []

    # -----------------------------------------------------
    # Evaluate questions
    # -----------------------------------------------------

    for i, question_data in enumerate(
        questions,
        start=1
    ):

        print("\n" + "-" * 80)

        print(
            f"[{i}/{len(questions)}] "
            f"{question_data['question']}"
        )

        result = evaluate_question(
            engine,
            question_data
        )

        results.append(result)

        # -------------------------------------------------
        # SAVE IMMEDIATELY
        # -------------------------------------------------

        save_results(results)

        # -------------------------------------------------
        # Display result
        # -------------------------------------------------

        print(
            f"Expected : "
            f"{result['expected_answerable']}"
        )

        print(
            f"Predicted: "
            f"{result['predicted_answerable']}"
        )

        print(
            f"Status   : "
            f"{result['evidence_status']}"
        )

        print(
            f"Score    : "
            f"{result['top_score']}"
        )

        print(
            f"Latency  : "
            f"{result['latency_seconds']} sec"
        )

        if result["error"]:

            print(
                "\n⚠ Question failed."
            )

            print(
                "Evaluation will continue."
            )

    # -----------------------------------------------------
    # Final metrics
    # -----------------------------------------------------

    metrics = calculate_metrics(
        results
    )

    print("\n" + "=" * 80)
    print("EVALUATION METRICS")
    print("=" * 80)

    for key, value in metrics.items():

        print(
            f"{key:30}: {value}"
        )

    print("\n" + "=" * 80)

    print(
        f"Results saved to: "
        f"{RESULTS_FILE}"
    )


if __name__ == "__main__":

    main()