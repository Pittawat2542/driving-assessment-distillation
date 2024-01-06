import os
from typing import Annotated

from dotenv import load_dotenv
import pandas as pd
import typer
from typer import Option
from pathlib import Path

from pymilvus import connections, FieldSchema, CollectionSchema, DataType, Collection, utility

DIMENSION = 1536

app = typer.Typer()


@app.command()
def main(collection: Annotated[str, Option("--collection", "-c")] = "driving",
         data_path: Annotated[str, Option("--data-path", "-d")] = "./data/train_small.csv"):
    connections.connect(host=os.environ.get("MILVUS_HOST"), port=os.environ.get("MILVUS_PORT"))
    print(f"Connected to Milvus: {os.environ.get('MILVUS_HOST')}:{os.environ.get('MILVUS_PORT')}")

    if utility.has_collection(collection):
        print(f"Collection {collection} already exists, drop it first")
        utility.drop_collection(collection)

    fields = [
        FieldSchema(name='id', dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="town", dtype=DataType.VARCHAR, max_length=16),
        FieldSchema(name="section", dtype=DataType.INT64),
        FieldSchema(name="scenario", dtype=DataType.VARCHAR, max_length=48),
        FieldSchema(name="part", dtype=DataType.INT64),
        FieldSchema(name="objects", dtype=DataType.VARCHAR, max_length=4096),
        FieldSchema(name="controls", dtype=DataType.VARCHAR, max_length=4096),
        FieldSchema(name="location", dtype=DataType.VARCHAR, max_length=128),
        FieldSchema(name="situation", dtype=DataType.VARCHAR, max_length=1024),
        FieldSchema(name="risk", dtype=DataType.VARCHAR, max_length=16),
        FieldSchema(name="message", dtype=DataType.VARCHAR, max_length=4096),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=DIMENSION),
        FieldSchema(name="message_embedding", dtype=DataType.VARCHAR, max_length=65535)
    ]

    schema = CollectionSchema(fields)

    driving_collection = Collection(collection, schema, consistency_level="Strong")

    index_params = {
        'index_type': 'FLAT',
        'metric_type': 'L2',
        'params': {'nlist': 1024}
    }
    driving_collection.create_index(field_name="embedding", index_params=index_params)
    print(f"Created collection {collection}")

    df = pd.read_csv(Path(data_path))
    print(f"Loaded {len(df)} entities from CSV file")

    for index, row in df.iterrows():
        entities = row.to_dict()
        entities["embedding"] = [float(x) for x in
                                 entities["objs_ctrls_embedding"].replace('[', '').replace(']', '').split(",")]
        entities["message_embedding"] = entities.pop("message_embedding")
        entities["location"] = entities.pop("chatgpt_extracted_location")
        entities["situation"] = entities.pop("chatgpt_extracted_situation")
        entities["risk"] = entities.pop("chatgpt_extracted_risk")
        entities["message"] = entities.pop("chatgpt_extracted_message")
        entities.pop("objs_ctrls_embedding")
        entities.pop("chatgpt_raw_response")
        driving_collection.insert(entities)

    print(f"Inserted {len(df)} entities into collection {collection}")


if __name__ == "__main__":
    load_dotenv()
    app()
