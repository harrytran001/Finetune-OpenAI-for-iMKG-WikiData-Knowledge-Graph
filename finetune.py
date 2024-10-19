import os
import json
import time
import random
import logging
import tiktoken
import argparse


from typing import List
from openai import OpenAI
from dotenv import load_dotenv
from collections import defaultdict

from model import TrainData
from constants import (
    TARGET_EPOCHS,
    ALLOWED_MODELS,
    MIN_DEFAULT_EPOCHS,
    MAX_DEFAULT_EPOCHS,
    MIN_TARGET_EXAMPLES,
    MAX_TARGET_EXAMPLES,
    SPARQL_GENERATION_PROMPT,
    TRAINED_MODEL_ID_PATH,
    PROCESSED_TRAIN_DATA_PATH,
)

load_dotenv()


encoding = tiktoken.get_encoding("cl100k_base")

logging.basicConfig(level=logging.INFO, force=True)

logger = logging.getLogger(__name__)


def check_valid_model(model_name: str) -> int:
    """
    Check if model is allowed to be trained

    Return:
        - Max token allowed per message thread
    """
    if model_name not in ALLOWED_MODELS.keys():
        raise ValueError(
            f"{','.join(ALLOWED_MODELS.keys())} models are allowed to be trained!"
        )
    return ALLOWED_MODELS[model_name]


def get_args():
    parser = argparse.ArgumentParser(
        description="Finetune GPT-4o for QA task on iMKG movie knowledge graph."
    )
    parser.add_argument(
        "--path", type=str, required=True, help="Path to the train data file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-2024-08-06",
        help="Base model to be trained",
    )
    parser.add_argument(
        "--sample", type=int, default=5, help="Number of samples per quesiton type"
    )
    args = parser.parse_args()

    if not os.path.exists(args.path):
        raise FileExistsError(f"Error: The path does not exist - {args.path}")

    max_token_per_sample = check_valid_model(args.model)
    args.max_token_per_sample = max_token_per_sample
    return args


def prepare_dataset(orginal_dataset_path, sample_per_question_types) -> List[TrainData]:
    with open(orginal_dataset_path, "r", encoding="utf-8") as f:
        raw_dataset = json.load(f)
    f.close()
    grouped = defaultdict(list)
    for item in raw_dataset:
        grouped[item["question_type"]].append(item)

    sampled_dataset = []
    for question_type, items in grouped.items():
        sampled_dataset.extend(
            random.sample(items, min(sample_per_question_types, len(items)))
        )
    random.shuffle(sampled_dataset)
    raw_dataset = sampled_dataset

    train_data: List[TrainData] = []
    for raw_data_point in raw_dataset:
        item = {
            "messages": [
                {
                    "role": "system",
                    "content": SPARQL_GENERATION_PROMPT,
                },
                {"role": "user", "content": raw_data_point["question"]},
                {"role": "assistant", "content": raw_data_point["sparql"]},
            ]
        }
        train_data.append(item)
    train_dataset_text = "\n".join([json.dumps(item) for item in train_data])
    with open(PROCESSED_TRAIN_DATA_PATH, "w", encoding="utf-8") as f:
        f.write(train_dataset_text)
    f.close()
    return train_data, len(grouped)


def validate_train_dataset(train_dataset: List[TrainData]):
    format_errors = defaultdict(int)
    for ex in train_dataset:
        if not isinstance(ex, dict):
            format_errors["data_type"] += 1
            continue

        messages = ex.get("messages", None)
        if not messages:
            format_errors["missing_messages_list"] += 1
            continue

        for message in messages:
            if "role" not in message or "content" not in message:
                format_errors["message_missing_key"] += 1

            if any(
                k not in ("role", "content", "name", "function_call", "weight")
                for k in message
            ):
                format_errors["message_unrecognized_key"] += 1

            if message.get("role", None) not in (
                "system",
                "user",
                "assistant",
                "function",
            ):
                format_errors["unrecognized_role"] += 1

            content = message.get("content", None)
            function_call = message.get("function_call", None)

            if (not content and not function_call) or not isinstance(content, str):
                format_errors["missing_content"] += 1

        if not any(message.get("role", None) == "assistant" for message in messages):
            format_errors["example_missing_assistant_message"] += 1

    if format_errors:
        logger.error("Found errors:")
        error_messsages = []
        for k, v in format_errors.items():
            error_messsages.append(f"{k}: {v}")
        raise TypeError(error_messsages)


def num_tokens_from_messages(messages, tokens_per_message=3, tokens_per_name=1):
    num_tokens = 0
    for message in messages:
        num_tokens += tokens_per_message
        for key, value in message.items():
            num_tokens += len(encoding.encode(value))
            if key == "name":
                num_tokens += tokens_per_name
    num_tokens += 3
    return num_tokens


def num_assistant_tokens_from_messages(messages):
    num_tokens = 0
    for message in messages:
        if message["role"] == "assistant":
            num_tokens += len(encoding.encode(message["content"]))
    return num_tokens


def estimate_token_count_and_no_epochs(train_dataset, max_token_per_example: int):
    n_epochs = TARGET_EPOCHS
    n_missing_system = 0
    n_missing_user = 0
    n_messages = []
    convo_lens = []
    assistant_message_lens = []

    for ex in train_dataset:
        messages = ex["messages"]
        if not any(message["role"] == "system" for message in messages):
            n_missing_system += 1
        if not any(message["role"] == "user" for message in messages):
            n_missing_user += 1
        n_messages.append(len(messages))
        convo_lens.append(num_tokens_from_messages(messages))
        assistant_message_lens.append(num_assistant_tokens_from_messages(messages))

    # Check if any user or system message is missing
    missing_message_errors = []
    if n_missing_system != 0:
        missing_message_errors.append(
            f"Num examples missing system message: {n_missing_system}",
        )
    if n_missing_user != 0:
        missing_message_errors.append(
            f"Num examples missing user message: {n_missing_user}",
        )
    if missing_message_errors:
        raise ValueError("\n".join(missing_message_errors))

    # Check if any message thread is longer than OpenAI limit
    n_too_long = sum(l > max_token_per_example for l in convo_lens)
    if n_too_long:
        logger.info(
            f"{n_too_long} examples may be over the {max_token_per_example} token limit, they will be truncated during fine-tuning"
        )

    n_train_examples = len(train_dataset)
    if n_train_examples * TARGET_EPOCHS < MIN_TARGET_EXAMPLES:
        n_epochs = min(MAX_DEFAULT_EPOCHS, MIN_TARGET_EXAMPLES // n_train_examples)
    elif n_train_examples * TARGET_EPOCHS > MAX_TARGET_EXAMPLES:
        n_epochs = max(MIN_DEFAULT_EPOCHS, MAX_TARGET_EXAMPLES // n_train_examples)

    n_billing_tokens_in_dataset = sum(
        min(max_token_per_example, length) for length in convo_lens
    )
    logger.info(
        f"Dataset has ~{n_billing_tokens_in_dataset} tokens that will be charged for during training"
    )
    logger.info(f"By default, you'll train for {n_epochs} epochs on this dataset")
    logger.info(
        f"By default, you'll be charged for ~{n_epochs * n_billing_tokens_in_dataset} tokens"
    )

    return n_epochs


def finetune(model: str, n_epochs: int):
    client = OpenAI()

    logger.info("Uploading data file for training")
    file = client.files.create(
        file=open(PROCESSED_TRAIN_DATA_PATH, "rb"), purpose="fine-tune"
    )
    logger.info(f"Uploaded, file_id: {file.id}")

    logger.info("Creating training job")
    job = client.fine_tuning.jobs.create(
        hyperparameters={"n_epochs": n_epochs},
        training_file=file.id,
        model=model,
    )
    job_id = job.id
    logger.info(f"Job created, job_id: {job_id}")

    while True:
        job = client.fine_tuning.jobs.retrieve(fine_tuning_job_id=job_id)
        job_id = job.id

        if job.error.message != None:
            logger.error(job.error.message)
            break

        logger.info(f"Training status: {job.status}")
        if job.status == "succeeded" and job.fine_tuned_model:
            with open(TRAINED_MODEL_ID_PATH, "w", encoding="utf-8") as f:
                f.write(job.fine_tuned_model)
            f.close()
            break
        time.sleep(5)


if __name__ == "__main__":
    args = get_args()
    train_dataset_path = f"{os.getcwd()}/{args.path}"
    logger.info("Step 1. Prepare Dataset")
    train_dataset, no_question_types = prepare_dataset(
        orginal_dataset_path=train_dataset_path, sample_per_question_types=args.sample
    )
    logger.info(
        f"Train data length: {len(train_dataset)}. Number of question types: {no_question_types}"
    )
    logger.info("Step 2. Validate Dataset")
    validate_train_dataset(train_dataset=train_dataset)
    logger.info("No errors found")

    logger.info("Step 3. Estimate token count and number of epochs")
    n_epochs = estimate_token_count_and_no_epochs(
        train_dataset=train_dataset, max_token_per_example=args.max_token_per_sample
    )

    logger.info("Step 4. Finetuning model")
    finetune(model=args.model, n_epochs=n_epochs)
