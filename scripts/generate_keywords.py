import re
import json
import numpy as np
from pathlib import Path
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from datetime import datetime
from collections import Counter
from tqdm.auto import tqdm
import fire

from .utils import push_data_to_hub


# ## START: Set up the LLM dependencies
# model = ChatOpenAI(
#     model_name="gpt-4-0125-preview",
#     temperature=0,
#     max_tokens=4095,
#     model_kwargs=dict(
#         top_p=1,
#         frequency_penalty=0,
#         presence_penalty=0,
#     ),
# )

model = ChatGroq(
    model_name="llama3-70b-8192",
    temperature=0,
    max_tokens=4095,
    model_kwargs=dict(
        top_p=1,
        # frequency_penalty=0,
        # presence_penalty=0,
    ),
)

class KeywordCount(BaseModel):
    type: str = Field(description="One of `extracted` or `generated`")
    count: int = Field(description="Number of keyword generated corresponding to the given type.")


class Keywords(BaseModel):
    extracted: list[str] = Field(description="List of keywords extracted from the input.")
    generated: list[str] = Field(description="List of keywords generated relevant to but not in the input.")
    num: list[KeywordCount] = Field(description="Contains the information about the number of keywords generated for each keyword type")


class JsonOutputParser(JsonOutputParser):
    def get_format_instructions(self) -> str:
        schema = Keywords.schema()
        schema.pop("title", None)
        schema.pop("type", None)

        return f"""The output should be formatted as a JSON instance that conforms to the JSON schema below.\n\nHere is the output schema:
        \n```\n{json.dumps(schema)}\n```"""


json_parser = JsonOutputParser(pydantic_object=Keywords)
num_sentences = 20

prompt = PromptTemplate(
    template="""Generate {num_sentences} sentences that can be classified to contain the topics provided. Try not to use the exact names of the topics in the sentences. Each sentence can contain one or more topics. Sentences should range from simple to complex.\n\n{format_instructions}\n\nTopics: {topics}""",
    input_variables=["topics"],
    partial_variables={"format_instructions": json_parser.get_format_instructions(), "num_sentences": num_sentences},
)


def run_chain(topics, alpha, prompt, model, all_topics, topic_idx):
    topics = ";".join(topics)
    base_dir = Path("data/synthetic_data")

    chain = prompt | model
    message = chain.invoke(dict(topics=topics))

    now = int(datetime.now().timestamp() * 1000)
    fname = f"synthetic_data-{now}.json"

    raw_fname = base_dir / "raw" / model.model_name / f"r_{fname}"
    proc_fname = base_dir / "processed" / model.model_name / f"p_{fname}"

    raw_fname.parent.mkdir(parents=True, exist_ok=True)
    proc_fname.parent.mkdir(parents=True, exist_ok=True)

    # Store the message
    with open(raw_fname, "w") as f:
        f.write(message.json())

    try:
        alpha = process_content(message.content, proc_fname, alpha, all_topics, topic_idx)
    except Exception as e:
        print(f"Error processing message: {e}")
        print(message.content)

    return alpha

## END: Set up the LLM dependencies


## START: Define some functions
def save_parse(content):
    matches = re.findall(r'\{[^{}]*\}', content)
    entities = []

    for match in matches:
        try:
            entities.append(json.loads(match))
        except json.JSONDecodeError:
            continue

    return entities


def process_content(content, proc_fname, alpha, all_topics, topic_idx):
    N = len(topic_idx)

    parsed = []
    try:
        parsed = json_parser.parse(content)

        if not isinstance(parsed, list):
            if isinstance(parsed, dict):
                assert "data" in parsed, "The parsed JSON object should have a 'data' key if it is a dictionary"
                parsed = parsed["data"]
            else:
                raise OutputParserException(f"Parser returned {type(parsed)}")

        with open(proc_fname, "w") as f:
            json.dump(parsed, f, indent=2)
    except OutputParserException as e:
        try:
            parsed = save_parse(content)
            with open(proc_fname, "w") as f:
                json.dump(parsed, f, indent=2)
        except Exception as e:
            print(f"Tried parsing failed: {e}")
            print(content)
    except Exception as e:
        print(f"Error parsing message: {e}")
        print(content)

    try:
        topic_counts = Counter([topic for pl in parsed for topic in pl.get("topics", []) if topic in all_topics])
    except KeyError:
        print(f"Error parsing message: {parsed}")
        print(content)
        raise Exception("Error parsing message")

    post = np.zeros(N)
    for topic, count in topic_counts.most_common():
        post[topic_idx[topic]] += count

    post = np.abs(post - post.max())
    alpha = alpha + post

    return alpha


def load_alpha(all_topics, topic_idx):
    alpha = np.ones(len(all_topics))

    for f in Path("data/synthetic_data/raw").glob("*/*.json"):
        proc_fname = f.parent.parent.parent / "processed" / f.parent.name / f.name.replace("r_", "p_")

        with open(f) as f:
            raw = json.load(f)

        content = raw["content"]
        alpha = process_content(content, proc_fname, alpha, all_topics, topic_idx)

    return alpha
## END: Define some functions


def main():
    # Load the World Bank topics
    with open("data/wb_topics.json") as f:
        # We remove duplicates and sort the topics
        all_topics = sorted(set(json.load(f)["all_topics"]))

    # all_topics.append("Other")
    topic_idx = {topic: i for i, topic in enumerate(all_topics)}

    N = len(topic_idx)
    num_topics = 10

    alpha = load_alpha(all_topics, topic_idx)

    print("The alpha values are:", alpha)
    print("The alpha value for each topic is:")
    for i, a in enumerate(alpha):
        print(f"{all_topics[i]}: {a}")

    # Create a loop to create synthetic data
    for i in tqdm(range(1000), desc="Running chain"):
        # Choose a random number of topics to ensure diversity in topic combinations
        # probs = np.random.dirichlet(alpha)
        probs = alpha / alpha.sum()  # Normalize
        objects = np.random.choice(range(N), size=num_topics, p=probs, replace=False)

        # DO NOT USE THIS LINE unless you want to use the most underrepresented topics
        # This could bias the data?
        # objects = np.argsort(alpha)[-num_topics:]  # Choose the topics with the highest alpha values as these are the most underrepresented

        # Choose the topics
        topics = [all_topics[i] for i in objects]

        alpha = run_chain(topics, alpha, prompt, model, all_topics, topic_idx)

        if i % 10 == 0:
            push_data_to_hub("avsolatorio", "synthetic_data", "raw")
            push_data_to_hub("avsolatorio", "synthetic_data", "processed")


    print("Done!")

    print("The alpha values are:", alpha)
    print("The alpha value for each topic is:")
    for i, a in enumerate(alpha):
        print(f"{all_topics[i]}: {a}")


if __name__ == "__main__":
    # poetry run python -m scripts.generate_synthetic_topic_data
    fire.Fire(main)
