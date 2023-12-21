from pathlib import Path
from typing import Annotated

import torch
import typer
from loguru import logger
import numpy as np
from datasets import load_dataset, DatasetDict
from transformers import AutoTokenizer, Seq2SeqTrainingArguments, AutoModelForSeq2SeqLM, set_seed
from typer import Option

from src.custom_data_loader_trainer import TaskPrefixTrainer, TaskPrefixDataCollator

SEED = 42
TRAIN_DATASET_PATH = Path("../data/train_small.csv")
TEST_DATASET_PATH = Path("../data/test_small.csv")


def load_data():
    dataset = load_dataset("csv", data_files={"train": str(TRAIN_DATASET_PATH), "test": str(TEST_DATASET_PATH)})
    train_valid_datasets = dataset['train'].train_test_split(test_size=0.1, seed=SEED)
    datasets = DatasetDict({
        'train': train_valid_datasets['train'],
        'valid': train_valid_datasets['test'],
        'test': dataset['test'],
    })

    logger.info("Loaded dataset")
    return datasets


def tokenize_function(tokenizer):
    def prefix_tokenize_function(examples):
        input_template = "[predict] objects: {}, controls: {}"
        inputs = tokenizer(
            [input_template.format(obj, ctrl) for obj, ctrl in zip(examples['objects'], examples['controls'])],
            max_length=1024, truncation=True)
        explain_template = "[explain] objects: {}, controls: {}"
        explain_inputs = tokenizer(
            [explain_template.format(obj, ctrl) for obj, ctrl in zip(examples['objects'], examples['controls'])],
            max_length=1024, truncation=True)
        inputs['explain_input_ids'] = explain_inputs['input_ids']
        inputs['explain_attention_mask'] = explain_inputs['attention_mask']

        with tokenizer.as_target_tokenizer():
            label_output_encodings = tokenizer(examples['chatgpt_extracted_message'], max_length=1024, truncation=True)

            reason_template = "location: {}, situation: {}, risk: {}"
            reason_output_encodings = tokenizer(
                [reason_template.format(location, situation, risk) for location, situation, risk in
                 zip(examples['chatgpt_extracted_location'], examples['chatgpt_extracted_situation'],
                     examples['chatgpt_extracted_risk'])], max_length=1024, truncation=True)

        inputs['labels'] = label_output_encodings['input_ids']
        inputs['reason_labels'] = reason_output_encodings['input_ids']

        return inputs

    return prefix_tokenize_function


def metrics(tokenizer):
    def compute_metrics(evaluation_predictions):
        predictions, labels = evaluation_predictions
        decoded_predictions = tokenizer.batch_decode(predictions[0], skip_special_tokens=True)

        labels = np.where(labels[0] != -100, labels[0], tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        predictions = np.array([prediction for prediction in decoded_predictions])
        labels = np.array([label for label in decoded_labels])

        acc = np.mean(predictions == labels)

        return {'accuracy': acc}

    return compute_metrics


app = typer.Typer()


@app.command()
def main(base_model: Annotated[str, Option("--model", "-m")] = "google/flan-t5-base",
         max_train_steps: Annotated[int, Option("--max-train-steps", "-t-steps")] = 10000,
         learning_rate: Annotated[float, Option("--learning-rate", "-lr")] = 5e-5,
         batch_size: Annotated[int, Option("--batch-size", "-b")] = 64,
         alpha: Annotated[float, Option("--alpha", "-a")] = 0.5, ):
    driving_datasets = load_data()
    tokenizer = AutoTokenizer.from_pretrained(base_model, return_tensors="pt")
    compute_metrics = metrics(tokenizer)
    tokenized_datasets = driving_datasets.map(tokenize_function(tokenizer), batched=True,
                                              remove_columns=['town', 'section', 'scenario', 'part', 'objects',
                                                              'controls', 'chatgpt_raw_response',
                                                              'chatgpt_extracted_location',
                                                              'chatgpt_extracted_situation', 'chatgpt_extracted_risk',
                                                              'chatgpt_extracted_message', 'objs_ctrls_embedding',
                                                              'message_embedding'])
    logger.info("Tokenized datasets")
    model = AutoModelForSeq2SeqLM.from_pretrained(base_model)
    logger.info("Loaded model")
    data_collator = TaskPrefixDataCollator(tokenizer=tokenizer, model=model)

    args = Seq2SeqTrainingArguments(
        "driving-distillation",
        remove_unused_columns=False,
        evaluation_strategy='steps',
        eval_steps=256,
        save_strategy='epoch',
        save_steps=256,
        logging_dir='logs',
        logging_strategy='steps',
        logging_steps=256,
        max_steps=max_train_steps,
        learning_rate=learning_rate,
        gradient_accumulation_steps=1,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        predict_with_generate=True,
        local_rank=-1,
        bf16=torch.cuda.is_available(),
        generation_max_length=512,
        prediction_loss_only=False,
    )

    trainer_kwargs = {
        'alpha': alpha,
        'output_rationale': True,
        'model': model,
        'args': args,
        'train_dataset': tokenized_datasets['train'],
        'eval_dataset': {'test': tokenized_datasets['test'], },
        'data_collator': data_collator,
        'tokenizer': tokenizer,
        'compute_metrics': compute_metrics,
    }

    trainer = TaskPrefixTrainer(
        **trainer_kwargs
    )

    logger.info("Starting training")
    trainer.train()
    logger.info("Finished training")


if __name__ == "__main__":
    Path("logs").mkdir(parents=True, exist_ok=True)
    logger.add("logs/{time}.log")
    set_seed(SEED)
    app()
