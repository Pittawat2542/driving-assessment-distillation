import os
import json
from pathlib import Path
from loguru import logger
from time import perf_counter
import pandas as pd
from scipy.spatial.distance import cosine, euclidean, cityblock, chebyshev
from pymilvus import connections, Collection
from tqdm import tqdm
from typing import Annotated
import typer
from typer import Option
from dotenv import load_dotenv

TEST_DATASET_PATH = Path("../data/test_small.csv")

app = typer.Typer()


def string_to_vector(s):
    return [float(x) for x in s.replace('[', '').replace(']', '').split(",")]


def search(query, collection):
    search_params = {
        "metric_type": "L2"
    }

    results = collection.search(
        data=[query],
        anns_field="embedding",
        param=search_params,
        limit=1,
        output_fields=["message", "message_embedding"]
    )

    return results


@app.command()
def main(collection: Annotated[str, Option("--collection", "-c")] = "driving",
         output_path: Annotated[str, Option("--output-path", "-o")] = "vectordb"):
    connections.connect(host=os.environ.get("MILVUS_HOST"), port=os.environ.get("MILVUS_PORT"))
    driving_collection = Collection(collection)
    driving_collection.load()
    logger.info(f"Collection: {collection}")

    test_df = pd.read_csv(TEST_DATASET_PATH)
    logger.info(f"Dataset: {TEST_DATASET_PATH}")
    output_path = Path(f"outputs/{output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output path: {output_path}")

    for index, row in tqdm(test_df.iterrows(), total=len(test_df)):
        if output_path.joinpath(f"{index}.json").exists():
            continue

        start_time = perf_counter()
        test_results = search(
            string_to_vector(row['objs_ctrls_embedding']),
            driving_collection)
        end_time = perf_counter()
        top1 = test_results[0][0]
        logger.info(f"Retrieved message: {top1.message}")
        logger.info(f"Time taken: {end_time - start_time} seconds")

        retrieved_embedding = string_to_vector(top1.message_embedding)
        gt_embedding = string_to_vector(row['message_embedding'])
        vector_db_distance = top1.distance
        cosine_distance = cosine(retrieved_embedding, gt_embedding)
        euclidean_distance = euclidean(retrieved_embedding, gt_embedding)
        cityblock_distance = cityblock(retrieved_embedding, gt_embedding)
        chebyshev_distance = chebyshev(retrieved_embedding, gt_embedding)

        result_obj = {
            "objects": row["objects"],
            "controls": row["controls"],
            "message": top1.message,
            "gt_message": row['chatgpt_extracted_message'],
            "vector_db_distance": vector_db_distance,
            "cosine": cosine_distance,
            "euclidean": euclidean_distance,
            "cityblock": cityblock_distance,
            "chebyshev": chebyshev_distance,
            "retrieved_embedding": retrieved_embedding,
            "gt_embedding": gt_embedding,
            "time": end_time - start_time,
            "created_at": pd.Timestamp.now().isoformat(),
        }

        logger.info(f"Vector DB distance: {vector_db_distance}")
        logger.info(f"Cosine: {cosine_distance}")
        logger.info(f"Euclidean: {euclidean_distance}")
        logger.info(f"Cityblock: {cityblock_distance}")
        logger.info(f"Chebyshev: {chebyshev_distance}")

        json.dump(result_obj, output_path.joinpath(f"{index}.json").open("w"))
        logger.info(f"Saved result for {index}")


if __name__ == '__main__':
    load_dotenv()
    Path("logs").mkdir(parents=True, exist_ok=True)
    logger.add("logs/{time}.log")
    app()
