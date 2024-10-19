# Finetune OpenAI for iMKG WikiData Knowledge Graph

## 1. Overview
This project fine-tunes OpenAI models for SPARQL generation and answering questions based on iMKG WikiData knowledge graph.

## 2. Prerequisites
- **Python 3.10+**
- **Poetry** for dependency management

## 3. Setup
### 3.1 **Clone the Repository**
```bash
git clone https://github.com/harrytran001/Finetune-OpenAI-for-iMKG-WikiData-Knowledge-Graph.git
cd Finetune-OpenAI-for-iMKG-WikiData-Knowledge-Graph
```

### 3.2 **Install dependencies**
```bash
poetry init
poetry env use 3.10
poetry shell
poetry install
```

### 3.3 **Environment Variable**
* Provide **OpenAI API Key** in .env
* You could refer to `.env.example` 

### 3.4 **Prepare train and test data**
* iMKG.ttl: Knowledge Graph File
* qa_train.json: Train dataset
* qa_test.json: Test dataset

These 3 files can be found [here](https://github.com/harrytran001/Finetune-OpenAI-for-iMKG-WikiData-Knowledge-Graph/releases/tag/Files). Place all files in the same directory as the repo.


## 4. Finetuning
```bash
python3 finetune --path qa_train.json
```
**Flags**:
* --path **required**: Path to your train data file
* --model **optional**: Base model to be trained. Default to `gpt-4o-2024-08-06`
* --sample **optional**: The number of data points per each question type. Default to `5`

Trained model name will be stored at `trained_model_id.txt`

## 5. Inference
`inference.ipynb` is used to compared the query result of generated SparQL query and that of sample query.
**Flow**:
* Pick a random data point from test data
* Generate a SparQL query with the finetuned model.
* Query results with the generated SparQL query
* Query results with the test SparQL query
* Compare results of two queries