import json

import torch
from typing import Annotated
from time import perf_counter
import typer
from loguru import logger
from typer import Option
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, set_seed

from evaluate.prompt import PROMPT_TEMPLATE

SEED = 42
TEST_DATASET_PATH = Path("../data/test_small.csv")
BASE_MODEL_NAME = "google/flan-t5-base"
DISTILLED_MODEL_PATH = str(Path("../distillation/checkpoint-10000"))

app = typer.Typer()


@app.command()
def main(model: Annotated[str, Option("--model", "-m")] = "base"):
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"

    if model == "base":
        model_name = BASE_MODEL_NAME
    elif model == "distilled":
        model_name = DISTILLED_MODEL_PATH
    else:
        raise ValueError("Unknown model name")

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    logger.info(f"Model: {model_name}")
    output_path = Path(f"outputs/{model_name.split('/')[-1]}")
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output path: {output_path}")

    dataset = pd.read_csv(TEST_DATASET_PATH)
    logger.info(f"Dataset: {TEST_DATASET_PATH}")

    for index, row in tqdm(dataset.iterrows(), total=len(dataset)):
        if output_path.joinpath(f"{index}.txt").exists():
            continue

        prompt = PROMPT_TEMPLATE.format(row["objects"], row["controls"])
        start_time = perf_counter()
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        outputs = model.generate(**inputs, max_length=1024)
        generated_message = tokenizer.decode(outputs[0], skip_special_tokens=True)
        end_time = perf_counter()
        logger.info(f"Generated message: {generated_message}")
        logger.info(f"Time taken: {end_time - start_time} seconds")

        result_obj = {
            "objects": row["objects"],
            "controls": row["controls"],
            "prompt": prompt,
            "message": generated_message,
            "time": end_time - start_time,
            "created_at": pd.Timestamp.now().isoformat(),
        }
        json.dump(result_obj, output_path.joinpath(f"{index}.json").open("w"))
        logger.info(f"Saved result for {index}")


if __name__ == "__main__":
    Path("logs").mkdir(parents=True, exist_ok=True)
    logger.add("logs/{time}.log")
    set_seed(SEED)
    app()
