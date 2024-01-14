import json

from typing import Annotated
from time import perf_counter
import typer
from dotenv import load_dotenv
from loguru import logger
from openai import OpenAI
from typer import Option
import pandas as pd
from pathlib import Path
from tqdm import tqdm

TEST_DATASET_PATH = Path("../data/test_small.csv")
BASE_MODEL_NAME = "google/flan-t5-base"
DISTILLED_MODEL_PATH = str(Path("../distillation/checkpoint-10000"))

app = typer.Typer()


def get_embedding(text, client, model="text-embedding-ada-002"):
    txt = text.replace("\n", " ")
    return client.embeddings.create(input=[txt], model=model).data[0].embedding


@app.command()
def main(model: Annotated[str, Option("--model", "-m")] = "base"):
    if model == "base":
        model_name = BASE_MODEL_NAME
    elif model == "distilled":
        model_name = DISTILLED_MODEL_PATH
    else:
        raise ValueError("Unknown model name")

    client = OpenAI()

    logger.info(f"Model: {model_name}")
    generated_message_path = Path(f"outputs/{model_name.split('/')[-1]}")
    output_path = Path(f"outputs/embeddings/{model_name.split('/')[-1]}")
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output path: {output_path}")

    dataset = pd.read_csv(TEST_DATASET_PATH)
    logger.info(f"Dataset: {TEST_DATASET_PATH}")

    for index, row in tqdm(dataset.iterrows(), total=len(dataset)):
        if output_path.joinpath(f"{index}.json").exists():
            continue

        generated_message_obj = json.load(
            generated_message_path.joinpath(f"{index}.json").open()
        )
        generated_message = generated_message_obj["message"]
        generated_message_time = generated_message_obj["time"]
        start_time = perf_counter()
        embedding = get_embedding(generated_message, client)
        end_time = perf_counter()
        logger.info(f"Time taken: {end_time - start_time} seconds")

        result_obj = {
            "objects": row["objects"],
            "controls": row["controls"],
            "message": generated_message,
            "embedding": embedding,
            "time": generated_message_time,
            "embedding_time": end_time - start_time,
            "created_at": pd.Timestamp.now().isoformat(),
        }
        json.dump(result_obj, output_path.joinpath(f"{index}.json").open("w"))
        logger.info(f"Saved result for {index}")


if __name__ == "__main__":
    Path("logs").mkdir(parents=True, exist_ok=True)
    logger.add("logs/{time}.log")
    load_dotenv()
    app()
