import json
from time import perf_counter, sleep
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, OpenAIError
from tiktoken import encoding_for_model
from rich.progress import track
from loguru import logger

MODEL = "gpt-3.5-turbo-1106"
RESULTS_PATH = Path("results")
LOGS_PATH = Path("logs")
TRAINING_PATH = Path("../data-preparation-scripts/training.csv")
PROMPT_TEMPLATE = """You're monitoring a driving assessment session. You're provided with real-time driving information including objects within 50 meters around the vehicle and the current vehicle state. The vehicle state has 4 elements:

1. throttle: a number between 0 and 0.7, where 0.7 indicates throttling
2. steering: a number in the range -1 to 1, where 0 indicates no steering
3. brake: a boolean, where True indicates that there is a brake
4. speed: current driving speed in km/h

Objects within 50 meters around the driver

Format
```
(distance between the driver and a target, a target: vehicle, pedestrian, stop sign, or traffic light)
```

Objects
```json
{}
```

Vehicle state
```json
{}
```

You are asked to give a comment on the current situation to support the driver by improving situation awareness and maintaining their high valence and medium arousal. The maximum length of comment is 10 words.

Following these steps to come up with a message:
 1. Think about the current location of the vehicle based on the provided information
 2. Think about the current situation of the vehicle based on the provided information
 3. Think if the current situation is safe or not. Does it possibly lead to hazardous events?
 4. Give a message to improve situation awareness and maintain their high valence and medium arousal
 
Output format
```json
{{
  "location": the current location,
  "situation": description of the current situation,
  "risk": risk level ("very high", "high", "medium", "low", "very low"),
  “message”: message
}}
```"""


def chatgpt(user_prompt: str) -> str:
    enc = encoding_for_model(MODEL)
    client = OpenAI(timeout=30, max_retries=5)

    logger.debug(f"Prompt: {user_prompt}")
    logger.info(f"Prompt length: {len(enc.encode(user_prompt))}")
    input_count = len(enc.encode(user_prompt))
    logger.info(f"Input count: {input_count}")
    logger.debug("Starting chat completion using OpenAI API")
    start_time = perf_counter()
    try:
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": user_prompt}],
            model=MODEL,
            temperature=0,
            seed=42,
        )
        end_time = perf_counter()
        logger.debug("Chat completion finished")
    except RateLimitError as e:
        print(e)
        raise e
    except OpenAIError as e:
        logger.error(f"Chat completion failed: {e}")
        raise e
    response = chat_completion.choices[0].message.content
    logger.debug(f"Response: {response}")
    output_count = len(enc.encode(response))
    logger.info(f"Output count: {output_count}")
    logger.info(f"Time taken: {end_time - start_time} seconds")
    if end_time - start_time < 0.4:
        sleep(0.4 - (end_time - start_time))
    return response


def main():
    df = pd.read_csv(TRAINING_PATH)
    for index, row in track(df.iterrows()):
        logger.info(f"Processing row {index}")
        with open("track.json", "r") as f:
            track_obj = json.load(f)
            if (RESULTS_PATH / f"{index}.txt").exists() and not track_obj["is_processing"]:
                logger.info(f"Skipping row {index}")
                continue
            track_obj["current"] = index
            track_obj["total"] = len(df)
            track_obj["is_processing"] = True
            with open("track.json", "w") as f:
                json.dump(track_obj, f)

        p = PROMPT_TEMPLATE.format("{ objects: " + row.objects + " }",
                                   "{ ctrl: " + row.controls.replace("[", "").replace("]", "") + " }")
        response = chatgpt(p)
        with open(RESULTS_PATH / f"{index}.txt", "w") as f:
            f.write(response)

        logger.info(f"Finished processing row {index}")
        with open("track.json", "r") as f:
            track_obj = json.load(f)
            track_obj["is_processing"] = False
            with open("track.json", "w") as f:
                json.dump(track_obj, f)


if __name__ == "__main__":
    LOGS_PATH.mkdir(exist_ok=True)
    RESULTS_PATH.mkdir(exist_ok=True)
    logger.add("logs/{time}.log")
    load_dotenv()

    if not Path("track.json").exists():
        track_json = {
            "current": 0,
            "total": 0,
            "is_processing": False,
        }
        with open("track.json", "w") as f:
            json.dump(track_json, f)
    main()
