import re
import json
import numpy as np
import pandas as pd
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

# from .utils import push_data_to_hub

# # ## START: Set up the LLM dependencies
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

# class KeywordCount(BaseModel):
#     type: str = Field(description="One of `extracted` or `generated`")
#     count: int = Field(description="Number of keyword generated corresponding to the given type.")

class Keywords(BaseModel):
    keywords: list[str] = Field(description="List of keywords extracted or related to the input.")
    num: int = Field(description="Number of the generated keywords.")

    # extracted: list[str] = Field(description="List of keywords extracted from the input.")
    # generated: list[str] = Field(description="List of keywords generated relevant to but not in the input.")
    # num: list[KeywordCount] = Field(description="Contains the information about the number of keywords generated for each keyword type")


class JsonOutputParser(JsonOutputParser):
    def get_format_instructions(self) -> str:
        schema = Keywords.schema()
        schema.pop("title", None)
        schema.pop("type", None)

        return f"""The output should be formatted as a JSON instance that conforms to the JSON schema below.\n\nHere is the output schema:
        \n```\n{json.dumps(schema)}\n```"""

    def strip_keys(schema):
        pass

json_parser = JsonOutputParser(pydantic_object=Keywords)
num_keywords = 20

prompt = PromptTemplate(
    template="""Please provide a total of at least {num_keywords} keywords related to the statistical indicator: {definition}\n\n{format_instructions}""",
    input_variables=["definition"],
    partial_variables={"format_instructions": json_parser.get_format_instructions(), "num_keywords": num_keywords},
)


def run_chain(data_group, idno, definition, prompt, model, rerun: bool = False):
    input_payload = dict(definition=definition)
    base_dir = Path(f"data/keywords/{data_group}")

    now = int(datetime.now().timestamp() * 1000)
    fname = f"{idno}-keywords-{now}.json"

    raw_dir = base_dir / "raw" / model.model_name
    proc_dir = base_dir / "processed" / model.model_name

    raw_dir.mkdir(parents=True, exist_ok=True)
    proc_dir.mkdir(parents=True, exist_ok=True)

    raw_fname = raw_dir / f"r_{fname}"
    proc_fname = proc_dir / f"p_{fname}"

    send_message = True
    content = None

    if not rerun:
        # Find raw data that may already be available.
        available = list(raw_fname.parent.glob(f"r_{idno}-keywords-*.json"))

        if len(available) > 0:
            raw_fname = available[-1]
            fname = raw_fname.name.replace("r_", "")
            proc_fname = proc_dir / f"p_{fname}"

            if not proc_fname.exists():
                message = json.loads(raw_fname.read_text())
                content = message["content"]

            send_message = False

    if send_message:
        chain = prompt | model
        message = chain.invoke(input_payload)

        # Store the message
        with open(raw_fname, "w") as f:
            f.write(message.json())

        content = message.content

    try:
        if content is not None:
            process_content(content, proc_fname)
    except Exception as e:
        print(f"Error processing message: {e}")
        print(content)

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


def process_content(content, proc_fname):
    parsed = {}

    try:
        parsed = json_parser.parse(content)

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

## END: Define some functions


def main(data_group: str, fname: str, rerun: bool = False):
    """
    data_group: Example: dhs, wdi, etc.
    fname: path to the file containing the idno and definition fields.
    """

    data = pd.read_csv(fname)
    data.columns = data.columns.str.lower()

    assert ("idno" in data.columns and "definition" in data.columns), "The expected data must have `idno` and `definition` fields."
    data = data.to_dict(orient="records")

    # Create a loop to create synthetic data
    for record in tqdm(data, desc="Running chain"):
        run_chain(data_group, record["idno"], record["definition"], prompt, model, rerun=rerun)

    print("Done!")

if __name__ == "__main__":
    # poetry run python -m scripts.generate_keywords --data_group=dhs --fname=~/Downloads/dhs_indic_csv.csv
    fire.Fire(main)
