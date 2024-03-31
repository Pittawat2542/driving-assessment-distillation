# Speed Up! Cost-Effective Large Language Model for ADAS Via Knowledge Distillation

This repository contains the code and datasets for the paper "Speed Up! Cost-Effective Large Language Model for ADAS Via Knowledge Distillation" accepted at [IEEE IV 2024](http://ieee-iv.org/2024/).

## Authors
Pittawat Taveekitworachai, Pratch Suntichaikul, Chakarida Nukoolkit, and Ruck Thawonmas

## Abstract

This paper presents a cost-effective approach to utilizing large language models (LLMs) as part of advanced driver-assistance systems (ADAS) through a knowledge-distilled model for driving assessment. LLMs have recently been employed across various domains. However, due to their size, they require sufficient computing infrastructure for deployment and ample time for generation. These characteristics make LLMs challenging to integrate into applications requiring real-time feedback, including ADAS. An existing study employed a vector database containing responses generated from an LLM to act as a surrogate model. However, this approach is limited when handling out-of-distribution (OOD) scenarios, which LLMs excel at. We propose a novel approach that utilizes a distilled model obtained from an established knowledge distillation technique to perform as a surrogate model for a target LLM, offering high resilience in handling OOD situations with substantially faster inference time. To assess the performance of the proposed approach, we also introduce a new dataset for driving scenarios and situations (\texttt{DriveSSD}), containing 124,248 records. Additionally, we augment randomly selected 12,425 records, 10\% of our \texttt{DriveSSD}, with text embeddings generated from an embedding model. We distill the model using 10,000 augmented records and test all approaches on the remaining 2,425 records. We find that the distilled model introduced in this study has better performance across metrics, with half of the inference time used by the previous approach. We make our source code and data publicly available.

## File structure
```
.
├── chatgpt-data-generation/ # 3. A script used to generate feedback and embeddings from ChatGPT for each data record.
├── data/ # 0. A folder containing all generated data, as well as all datasets used for training and evaluating the approaches.
├── data-generation-scripts/ # 1. Scripts are used to interact with CARLA, driving an agent around a map according to a given scenario, while collecting various data from sensors.
├── data-preparation-scripts/ # 2. Scripts used to preprocess the data generated from CARLA.
├── distillation/ # 6. A script used to perform model distillation given a dataset.
├── docker-compose.yml/ # 5.1. A Docker Compose file for spinning up the vector database.
├── evaluate/ # 7. Scripts for evaluation of all approaches and data analysis.
├── requirements.txt/ # 0. A Python dependencies list.
├── vector-database/ # 4. A script used to propagate the vector database.
└── volumes/ # 5.2. A vector database containing all of the data, along with the Docker Compose file.
```

## Installation and Usage
0. Create a virtual environment (if needed):
```bash
conda create -n driving-distill python=3.11
```
and activate it:
```bash
conda activate driving-distill
```
1. Copy `chatgpt-data-generation/.env.example` and rename it to `chatgpt-data-generation/.env.`. Follow instructions on [this page](https://platform.openai.com/docs/api-reference/authentication) to obtain your own OpenAI API key.
2. Install the requirements:
```bash
pip install -r requirements.txt
```
3. For Python file, run it by executing `python <filename>.py`. For Jupyter Notebook (`.ipynb`), open it with Jupyter Notebook and run it.
