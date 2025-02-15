# Separate file which contains the functions to convert predictions to hard
# labels and calculate the IoU score using the settings of our best model in
# SemEval 2025 Task 3.
import argparse
import collections
from scipy.stats import spearmanr

import jsonlines
import numpy as np
from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, TrainingArguments, Trainer

def add_answers_column(example):
    starts, texts = [], []
    for hard_label in example["hard_labels"]:
        starts.append(hard_label[0])
        texts.append(example["context"][hard_label[0]:hard_label[1]])
    example["answers"] = {"answer_start": starts, "text": texts}
    return example

def to_dataset(file_path):
    mushroom = load_dataset("json", data_files=file_path)["train"]
    mushroom = mushroom.rename_column("model_output_text", "context")
    mushroom = mushroom.rename_column("model_input", "question")
    if "hard_labels" in mushroom.column_names:
        mushroom = mushroom.map(add_answers_column)
    else:
        print("No hard labels found in the evaluation data: only generating predictions.")

    return mushroom

def preprocess_examples(examples, tokenizer):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=384,
        truncation="only_second",
        stride=128,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ]

    inputs["example_id"] = example_ids
    return inputs


def score_iou(ref_dict, pred_dict):
    """
    Computes intersection-over-union between reference and predicted hard
    labels, for a single datapoint.

    Arguments:
        ref_dict (dict): a gold reference datapoint,
        pred_dict (dict): a model's prediction

    Returns:
        int: The IoU, or 1.0 if neither the reference nor the prediction contain hallucinations
    """
    # ensure the prediction is correctly matched to its reference
    assert ref_dict['id'] == pred_dict['id']
    # convert annotations to sets of indices
    ref_indices = {idx for span in ref_dict['hard_labels'] for idx in range(*span)}
    pred_indices = {idx for span in pred_dict['hard_labels'] for idx in range(*span)}
    # avoid division by zero
    if not pred_indices and not ref_indices: return 1.
    # otherwise compute & return IoU
    return len(ref_indices & pred_indices) / len(ref_indices | pred_indices)

def score_cor(ref_dict, pred_dict):
    """computes Spearman correlation between predicted and reference soft labels, for a single datapoint.
    inputs:
    - ref_dict: a gold reference datapoint,
    - pred_dict: a model's prediction
    returns:
    the Spearman correlation, or a binarized exact match (0.0 or 1.0) if the reference or prediction contains no variation
    """
    # ensure the prediction is correctly matched to its reference
    assert ref_dict['id'] == pred_dict['id']
    # convert annotations to vectors of observations
    ref_vec = [0.] * ref_dict['text_len']
    pred_vec = [0.] * ref_dict['text_len']
    for span in ref_dict['soft_labels']:
        for idx in range(span['start'], span['end']):
            ref_vec[idx] = span['prob']
    for span in pred_dict['soft_labels']:
        for idx in range(span['start'], span['end']):
            pred_vec[idx] = span['prob']
    # constant series (i.e., no hallucination) => cor is undef
    if len({round(flt, 8) for flt in pred_vec}) == 1 or len({round(flt, 8) for flt in ref_vec}) == 1 :
        return float(len({round(flt, 8) for flt in ref_vec}) == len({round(flt, 8) for flt in pred_vec}))
    # otherwise compute Spearman's rho
    return spearmanr(ref_vec, pred_vec).correlation

def infer_soft_labels(hard_labels):
    """reformat hard labels into soft labels with prob 1"""
    return [
        {
            'start': start,
            'end': end,
            'prob': 1.0,
        }
        for start, end in hard_labels
    ]

def find_possible_spans(answers, example):
    """
    Creates and filters possible hallucination spans.

    Arguments:
        answers (list): List containing dictionaries with spans as text and
                        logit scores.
        example: The instance which is being predicted. The context is used to map the predicted text to the start
                 and end indexes of the target context.
    Returns:
        list: List with lists of hard labels.
    """
    best_answer = max(answers, key=lambda x: x["logit_score"])
    threshold = best_answer["logit_score"] * 0.8
    hard_labels = []
    for answer in answers:
        if answer["logit_score"] > threshold:
            start_index = example["context"].index(answer["text"])
            end_index = start_index + len(answer["text"])
            hard_labels.append([start_index, end_index])
    soft_labels = infer_soft_labels(hard_labels)
    return hard_labels, soft_labels

def compute_metrics(start_logits, end_logits, features, examples, predictions_file):
    """
    Function to process predictions, create spans and if possible,
    calculates IoU

    Arguments:
        args (ArgumentParser): Arguments supplied by user.
        start_logits (list): Logits of all start positions.
        end_logits (list): Logits of all end positions.
        features (Dataset): Dataset containing features of questions and context.
        examples (Dataset): Dataset containing examples with hard labels.

    Returns:
        None
    """
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in tqdm(examples):
        example_id = example["id"]
        context = example["context"]
        answers = []

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1: -20 - 1: -1].tolist()
            end_indexes = np.argsort(end_logit)[-1: -20 - 1: -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (
                            end_index < start_index
                            or end_index - start_index + 1 > 30
                    ):
                        continue

                    answer = {
                        "text": context[offsets[start_index][0]: offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)

        # Select the answer with the best score
        if len(answers) > 0:
            hard_labels, soft_labels = find_possible_spans(answers, example)
            predicted_answers.append(
                {"id": example_id, "hard_labels": hard_labels, "soft_labels": soft_labels}
            )
        else:
            predicted_answers.append({"id": example_id, "hard_labels": [], "soft_labels": []})

    with jsonlines.open(predictions_file, mode="w") as writer:
        writer.write_all(predicted_answers)

    if "answers" in examples.column_names:
        true_answers = [{"id": ex["id"], "hard_labels": ex["hard_labels"], "soft_labels": ex["soft_labels"],
                         "text_len": len(ex["context"])} for ex in examples]
        ious = np.array([score_iou(r, d) for r, d in zip(true_answers, predicted_answers)])
        cors = np.array([score_cor(r, d) for r, d in zip(true_answers, predicted_answers)])

        print(f"IOU: {ious.mean():.8f}, COR: {cors.mean():.8f}")
    else:
        print("Evaluation data contained no answers. No scores to show.")

def main(model_path, evaluation_file_path, output_file):
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_path
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path
    )
    # Initialize Trainer
    args = TrainingArguments(
        output_dir="output_dir",
        per_device_eval_batch_size=16,
        report_to="none"
    )

    model = Trainer(
        model=model,
        args=args,
        tokenizer=tokenizer,
    )

    mushroom_dataset = to_dataset(evaluation_file_path)
    features = mushroom_dataset.map(
        preprocess_examples,
        batched=True,
        remove_columns=mushroom_dataset.column_names,
        fn_kwargs={"tokenizer": tokenizer}
    )

    predictions, _, _ = model.predict(features)
    start_logits, end_logits = predictions
    compute_metrics(start_logits, end_logits, features, mushroom_dataset, output_file)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('model_name', type=str)
    p.add_argument('evaluation_file_path', type=str)
    p.add_argument('output_file', type=str)
    a = p.parse_args()
    main(a.model_name, a.evaluation_file_path, a.output_file)
