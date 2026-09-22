# Text-to-SQL with a Small Language Model

An academic project exploring how a small language model translates natural-language geography questions into executable SQL. The main artifact is [`llm_for_sql.ipynb`](llm_for_sql.ipynb).

The notebook compares three approaches on GeoQuery:

1. Zero-shot and few-shot prompting
2. LoRA fine-tuning of `HuggingFaceTB/SmolLM2-360M-Instruct`
3. Context-free grammar constrained decoding with `xgrammar`

It evaluates generated SQL by executing queries against a SQLite geography database and reporting precision, recall, F1, exact match, and grammatical validity.

## Run

Open the notebook in Google Colab or Kaggle with a CUDA GPU, then run its cells in order. The notebook installs its Python dependencies and downloads the GeoQuery data and database. A CPU runtime is not practical for the fine-tuning experiment.

## Data

The GeoQuery SQL dataset and geography database come from [text2sql-data](https://github.com/jkkummerfeld/text2sql-data). The split used here contains 549 training, 49 development, and 279 test examples.
