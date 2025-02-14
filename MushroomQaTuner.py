import argparse
import json
import jsonlines
import logging
import os

from datasets import load_dataset
from transformers import AutoTokenizer, DefaultDataCollator, AutoModelForQuestionAnswering
import numpy as np
from transformers import Trainer, TrainingArguments
import collections
from tqdm.auto import tqdm

logging.basicConfig(filename='results.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s')

class MushroomQaTuner:
    def __init__(self, args):
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        self.model = AutoModelForQuestionAnswering.from_pretrained(args.model_name)
        self.data_collator = DefaultDataCollator()

        self.max_length = args.max_length
        self.stride = args.stride
        self.n_best = args.n_best
        self.max_answer_length = args.max_answer_length

        mushroom_train = self.to_dataset(args.train_file)
        mushroom_val = self.to_dataset(args.val_file, train=False)
        mushroom_test =  self.to_dataset(args.test_file, train=False)

        train_dataset = mushroom_train.map(
            self.preprocess_training_examples,
            batched=True,
            remove_columns=mushroom_train.column_names,
        )

        validation_dataset = mushroom_val.map(
            self.preprocess_validation_examples,
            batched=True,
            remove_columns=mushroom_val.column_names,
        )

        test_dataset = mushroom_test.map(
            self.preprocess_validation_examples,
            batched=True,
            remove_columns=mushroom_test.column_names,
        )

        output_dir = f"/model/mushroom_qa_{args.model_name.split('/')[1]}"

        if args.model_output_dir:
            output_dir = args.model_output_dir

        training_args = TrainingArguments(
            output_dir=output_dir,
            eval_strategy="epoch",
            learning_rate=args.learning_rate,
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=args.batch_size,
            num_train_epochs=args.epochs,
            weight_decay=args.weight_decay,
            push_to_hub=False,
            report_to="none",
        )

        # Add compute_metrics function
        def compute_metrics(eval_pred):
            start_logits, end_logits = eval_pred.predictions
            features = validation_dataset
            examples = mushroom_val
            return self.compute_metrics(args, start_logits, end_logits, features, examples)

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=validation_dataset,
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics,
        )

        trainer.train()

        trainer.save_model(output_dir)

        predictions, _, _ = trainer.predict(validation_dataset)
        start_logits, end_logits = predictions
        self.compute_metrics(args, start_logits, end_logits, validation_dataset, mushroom_val)

        all_args = " \\\n".join([f" --{key}={value}" for key, value in vars(args).items() if value])
        self.log_and_print(f"Used settings:\n{all_args}", False)

    def log_and_print(self, message, printed=True):
        """Logs a message and prints it to the console."""
        logging.info(message)
        if printed:
            print(message)

    def add_answers_column(self, example):
        starts, texts = [], []
        for hard_label in example["hard_labels"]:
            starts.append(hard_label[0])
            texts.append(example["context"][hard_label[0]:hard_label[1]])
        example["answers"] = {"answer_start": starts, "text": texts}
        return example

    def has_answer_start(self, example):
        """Checks if an example has a valid answer_start."""
        return len(example["answers"]["answer_start"]) > 0 and example["answers"]["answer_start"][0] != -1

    def annotate_and_save(self, file_path):
        """
        Read a single dataset, annotate it, and save the results.
        """
        with open(file_path, 'r') as file:
            data = [json.loads(line) for line in file]
        transformed = []
        id = 1
        for datapoint in data:
            question = datapoint["model_input"]
            context = datapoint["model_output_text"]
            answer_id = 1
            for hard_label in datapoint["hard_labels"]:
                start = hard_label[0]
                end = hard_label[1]
                data_dict = {"id": f"{id}-{answer_id}", "model_output_text": context, "model_input": question,
                             "answers": {
                                 "answer_start": [
                                     start], "text": [context[start:end]], "span": hard_label}}
                transformed.append(data_dict)
                answer_id += 1
        id += 1
        new_file_path = f"{os.path.splitext(file_path)[0]}.train.jsonl"
        print(new_file_path)
        with open(new_file_path, 'w') as file:
            for entry in transformed:
                file.write(json.dumps(entry) + '\n')
        return new_file_path

    def to_dataset(self, file_path, train=True):
        if train:
            file_path = self.annotate_and_save(file_path)
        mushroom = load_dataset("json", data_files=file_path)["train"]
        mushroom = mushroom.rename_column("model_output_text", "context")
        mushroom = mushroom.rename_column("model_input", "question")
        if not train:
            mushroom = mushroom.map(self.add_answers_column)
        # mushroom = mushroom.filter(has_answer_start)

        return mushroom

    def preprocess_training_examples(self, examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = self.tokenizer(
            questions,
            examples["context"],
            max_length=self.max_length,
            truncation="only_second",
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        offset_mapping = inputs.pop("offset_mapping")
        sample_map = inputs.pop("overflow_to_sample_mapping")
        answers = examples["answers"]
        start_positions = []
        end_positions = []

        for i, offset in enumerate(offset_mapping):
            sample_idx = sample_map[i]
            answer = answers[sample_idx]
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i)

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx
            while sequence_ids[idx] == 1:
                idx += 1
            context_end = idx - 1

            # If the answer is not fully inside the context, label is (0, 0)
            if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

        inputs["start_positions"] = start_positions
        inputs["end_positions"] = end_positions
        return inputs

    def preprocess_validation_examples(self, examples):
        questions = [q.strip() for q in examples["question"]]
        inputs = self.tokenizer(
            questions,
            examples["context"],
            max_length=self.max_length,
            truncation="only_second",
            stride=self.stride,
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

    def score_iou(self, ref_dict, pred_dict):
        """computes intersection-over-union between reference and predicted hard labels, for a single datapoint.
        inputs:
        - ref_dict: a gold reference datapoint,
        - pred_dict: a model's prediction
        returns:
        the IoU, or 1.0 if neither the reference nor the prediction contain hallucinations
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

    def find_possible_spans(self, answers, example):
        best_answer = max(answers, key=lambda x: x["logit_score"])
        threshold = best_answer["logit_score"] * 0.8
        hard_labels = []
        for answer in answers:
            if answer["logit_score"] > threshold:
                start_index = example["context"].index(answer["text"])
                end_index = start_index + len(answer["text"])
                hard_labels.append([start_index, end_index])
        return hard_labels

    def compute_metrics(self, args, start_logits, end_logits, features, examples):
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

                start_indexes = np.argsort(start_logit)[-1: -self.n_best - 1: -1].tolist()
                end_indexes = np.argsort(end_logit)[-1: -self.n_best - 1: -1].tolist()
                for start_index in start_indexes:
                    for end_index in end_indexes:
                        # Skip answers that are not fully in the context
                        if offsets[start_index] is None or offsets[end_index] is None:
                            continue
                        # Skip answers with a length that is either < 0 or > max_answer_length
                        if (
                                end_index < start_index
                                or end_index - start_index + 1 > self.max_answer_length
                        ):
                            continue

                        answer = {
                            "text": context[offsets[start_index][0]: offsets[end_index][1]],
                            "logit_score": start_logit[start_index] + end_logit[end_index],
                        }
                        answers.append(answer)

            # Select the answer with the best score
            if len(answers) > 0:
                hard_labels = self.find_possible_spans(answers, example)
                predicted_answers.append(
                    {"id": example_id, "hard_labels": hard_labels}
                )
            else:
                predicted_answers.append({"id": example_id, "hard_labels": []})

        with jsonlines.open(f'/data/{args.predictions_filename}.jsonl', mode='w') as writer:
            writer.write_all(predicted_answers)

        true_answers = [{"id": ex["id"], "hard_labels": ex["hard_labels"]} for ex in examples]
        ious = np.array([self.score_iou(r, d) for r, d in zip(true_answers, predicted_answers)])

        self.log_and_print(f"IOU: {ious.mean():.8f}")
        return {"iou": ious.mean()}



def create_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--train_file", default="/data/train/mushroom.en-train_LABELED.v1.jsonl",
                        type=str,
                        help="Input file to learn from")
    parser.add_argument("-d", "--val_file", type=str, default="/data/val/mushroom.en-val.v1.jsonl",
                        help="Separate dev set to read in")
    parser.add_argument("-t", "--test_file", type=str, default="data/val/mushroom.en-val.v1.jsonl",
                        help="If added, use trained model to predict on test set")
    parser.add_argument("-m", "--model_name", type=str, default="FacebookAI/roberta-base",
                        help="Model to use")
    parser.add_argument("-mo", "--model_output_dir", type=str, default=None,
                        help="Name to give the create model")
    parser.add_argument("-pn", "--predictions_filename", type=str, help="Name the predictions file")
    parser.add_argument("-ml", "--max_length", type=int, default=384)
    parser.add_argument("-s", "--stride", type=int, default=128)
    parser.add_argument("-nb", "--n_best", type=int, default=20)
    parser.add_argument("-mal", "--max_answer_length", type=int, default=30)
    parser.add_argument("-lr", "--learning_rate", type=float, default=2e-5)
    parser.add_argument("-bs", "--batch_size", type=int, default=16)
    parser.add_argument("-ep", "--epochs", type=int, default=3)
    parser.add_argument("-wd", "--weight_decay", type=float, default=0.1)

    args = parser.parse_args()
    return args

def main():
    args = create_arg_parser()
    MushroomQaTuner(args)


if __name__ == "__main__":
    main()