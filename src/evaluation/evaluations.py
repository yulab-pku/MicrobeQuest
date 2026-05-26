import re
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import json  # For reading/writing JSON files
from typing import List, Dict
import csv

def evaluate_results_from_file(input_path: str) -> List[Dict]:
    """
    Evaluate each record in a JSON file that contains predictions and references,
    then write the results with evaluation scores to a new file.
    Also write debug information including tokens to a separate file.
    """
    # Load original predictions
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    debug_info = []  # Used for storing debugging information

    # Evaluate each item
    for item in data:
        expected = item.get("expected_answer", "")
        predicted = item.get("predicted_answer", "")
        evaluation, debug_record = evaluate_pair(expected, predicted)
        item["evaluation"] = evaluation
        debug_info.append(debug_record)

    return data


def summarize_results(results: List[Dict]) -> Dict[str, float]:
    """
    Aggregate the evaluation results: compute average accuracy, recall, F1, and BLEU scores.

    Args:
        results: List of evaluation records, each containing an 'evaluation' field.

    Returns:
        A dictionary with average values for the main metrics.
    """
    if not results:
        print("No results to summarize.")
        return {}

    total = len(results)
    correct = sum(1 for r in results if r["evaluation"]["accuracy"] == 1.0)
    overall_accuracy = correct / total if total > 0 else 0.0

    total_recall = sum(r["evaluation"]["recall"] for r in results)
    total_f1 = sum(r["evaluation"]["f1_score"] for r in results)
    total_bleu = sum(r["evaluation"]["bleu_score"] for r in results)

    avg_recall = total_recall / total if total > 0 else 0.0
    avg_f1 = total_f1 / total if total > 0 else 0.0
    avg_bleu = total_bleu / total if total > 0 else 0.0
    #
    # print(f"Overall Accuracy: {overall_accuracy:.2%} ({correct}/{total})")
    # print(f"Average Recall:   {avg_recall:.4f}")
    # print(f"Average F1 Score: {avg_f1:.4f}")
    # print(f"Average BLEU:     {avg_bleu:.4f}")

    return {
        "accuracy": round(overall_accuracy, 4),
        "recall": round(avg_recall, 4),
        "f1_score": round(avg_f1, 4),
        "bleu_score": round(avg_bleu, 4)
    }
def extract_answer(text: str) -> str:
    """
    Extract <Answer>... from the text The content in <Answer> (if any), otherwise keep the original text
    """
    matches = re.findall(r'<Answer>\s*(.*?)\s*</Answer>', text, re.IGNORECASE | re.DOTALL)
    for match in matches:
        cleaned = match.strip()
        if cleaned:
            return cleaned

    fallback_match = re.search(r'<Answer>\s*(.*)', text, re.IGNORECASE)
    if fallback_match:
        return fallback_match.group(1).strip()

    return str(text).strip()
def clean_text(text: str) -> str:
    """
    Normalize text:
    - Convert to lowercase
    - Replace subscript/superscript digits with normal digits
    - Remove punctuation/symbols, keep only letters, numbers, and decimals

    Args:
        text: Raw text string

    Returns:
        Cleaned string
    """
    text = str(text)
    subscript_map = str.maketrans('₀₁₂₃₄₅₆₇₈₉', '0123456789')
    superscript_map = str.maketrans('⁰¹²³⁴⁵⁶⁷⁸⁹', '0123456789')

    text = text.lower()
    text = text.translate(subscript_map).translate(superscript_map)
    text = re.sub(r'[^a-z0-9.\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Dictionary of English chemical names and their molecular formulas
CHEMICAL_SYNONYMS = {
    "nitrogen": "n2",
    "oxygen": "o2",
    "hydrogen": "h2",
    "carbon dioxide": "co2",
    "carbon monoxide": "co",
    "methane": "ch4",
    "ammonia": "nh3",
    "water": "h2o",
}


def replace_synonyms(text: str) -> str:
    """
    Replace English chemical names with their molecular formulas.

    Args:
        text: Input string.

    Returns:
        Modified string with formulas.
    """
    for word, formula in CHEMICAL_SYNONYMS.items():
        text = re.sub(r'\b' + re.escape(word) + r'\b', formula, text)
    return text


def normalize_number(token: str) -> str:
    """
    Normalize number format:
    - Convert "7.0" to "7"
    - Keep "6.8" as is

    Args:
        token: Individual token (word or number)

    Returns:
        Normalized number token
    """
    try:
        if '.' in token:
            num = float(token)
            return str(int(num)) if num.is_integer() else str(num)
        return token
    except ValueError:
        return token


def preprocess(text: str) -> List[str]:
    """
    Full preprocessing pipeline:
    - Clean text
    - Replace chemical names
    - Tokenize
    - Normalize numbers

    Args:
        text: Raw input text

    Returns:
        List of preprocessed tokens
    """
    text = clean_text(text)
    text = replace_synonyms(text)
    return [normalize_number(t) for t in text.split()]


def evaluate_pair(expected: str, predicted: str) -> (Dict[str, float], Dict[str, any]):
    """
    Evaluate a single expected vs. predicted answer pair using multiple metrics.
    Also return a debug record including raw text and tokenized results.
    """
    expected_raw = extract_answer(expected)
    predicted_raw = extract_answer(predicted)
    expected_tokens = preprocess(expected_raw)
    predicted_tokens = preprocess(predicted_raw)

    expected_set = set(expected_tokens)
    predicted_set = set(predicted_tokens)
    intersection = expected_set & predicted_set

    recall = len(intersection) / len(expected_set) if expected_set else 0.0
    precision = len(intersection) / len(predicted_set) if predicted_set else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = 1.0 if expected_set == predicted_set else 0.0

    bleu = sentence_bleu(
        [expected_tokens],
        predicted_tokens,
        smoothing_function=SmoothingFunction().method1
    )

    overall_score = (accuracy + recall + f1 + bleu) / 4.0

    evaluation_result = {
        "accuracy": round(accuracy, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "bleu_score": round(bleu, 4),
        "overall_score": round(overall_score, 4)
    }

    debug_record = {
        "expected": expected_raw,
        "predicted": predicted_raw,
        "expected_tokens": expected_tokens,
        "predicted_tokens": predicted_tokens
    }

    return evaluation_result, debug_record
