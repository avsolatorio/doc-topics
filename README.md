# Doc Topics

The aim of this project is to leverage LLMs to understand document topics. The topics are adopted from the World Bank's topic classification: https://www.worldbank.org/en/topic.

## Steps

- Collect the topics from the World Bank's website: run `poetry run python scripts/get_wb_topics.py`
- Generate synthetic data using GPT-4: run `poetry run python scripts/generate_synthetic_topic_data.py`

