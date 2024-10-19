SPARQL_GENERATION_PROMPT = """You are a useful SparQL assistant. You are tasked to review a question and generate a SparQL to answer the question.
SparQL Database used is WikiData. [<text>] is topic entity in the question.
Only use these two prefixes PREFIX wd: <https://www.wikidata.org/entity/> and PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> if needed.
Do not use wdt syntax to query WikiData"""

TARGET_EPOCHS = 3
MIN_DEFAULT_EPOCHS = 1
MAX_DEFAULT_EPOCHS = 25
MIN_TARGET_EXAMPLES = 100
MAX_TARGET_EXAMPLES = 25000

ALLOWED_MODELS = {
    "gpt-4o-2024-08-06": 65536,
    "gpt-4o-mini-2024-07-18": 65536,
    "gpt-4-0613": 65536,
    "gpt-3.5-turbo-0125": 16385,
    "gpt-3.5-turbo-1106": 16385,
    "gpt-3.5-turbo-0613": 4096,
}

TEST_DATA_PATH = "qa_test.json"
PROCESSED_TRAIN_DATA_PATH="processed_train_data.jsonl"
TRAINED_MODEL_ID_PATH="trained_model_id.txt"
