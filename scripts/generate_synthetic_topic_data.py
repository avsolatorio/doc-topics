import os
import json
import numpy as np
from langchain_core.pydantic_v1 import BaseModel, Field
from openai import OpenAI
from langchain_core.output_parsers import JsonOutputParser
from typing import List

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI




client = OpenAI(api_key=os.environ['OPENAI_API_KEY'])

model = ChatOpenAI(
    model_name="gpt-4-0125-preview",
    temperature=1,
    max_tokens=4095,
    model_kwargs=dict(
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
    ),
)

class TopicSentence(BaseModel):
    topics: list[str] = Field(description="List of topics contained in the sentence")
    sentence: str = Field(description="The sentence containing the topics")

class JsonOutputParser(JsonOutputParser):
    def get_format_instructions(self) -> str:
        schema = TopicSentence.schema()
        schema.pop("title", None)
        schema.pop("type", None)

        return f"""The output should be formatted as a JSON instance that conforms to the JSON schema below.\n\nHere is the output schema:
        \n```\n{json.dumps(schema)}\n```"""


json_parser = JsonOutputParser(pydantic_object=TopicSentence)


# Load the World Bank topics

with open("data/wb_topics.json") as f:
    all_topics = json.load(f)["all_topics"]

# all_topics.append("Other")
N = len(all_topics)
alpha = np.ones(N)
num_topics = 10

instruction = ""


num_sentences = 20

prompt = PromptTemplate(
    template="""Generate {num_sentences} sentences that can be classified to contain the topics provided. Try not to use the exact names of the topics in the sentences. Each sentence can contain one or more topics. Ensure the topics are sampled uniformly. Sentences should range from simple to complex.\n\n{format_instructions}\n\nTopics: {topics}""",
    input_variables=["topics"],
    partial_variables={"format_instructions": json_parser.get_format_instructions(), "num_sentences": num_sentences},
)

chain = prompt | model | json_parser



def send_request(topics):
    topics = ";".join(topics)

    response = client.chat.completions.create(
        model="gpt-4-0125-preview",
        messages=[
            dict(role="system", content=instruction),
            dict(role="user", content=topics),
        ],
        temperature=1,
        max_tokens=4095,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    output = response.model_dump()

    return output


# Create a loop to create synthetic data
for i in range(100):
    # Choose a random number of topics
    probs = np.random.dirichlet(alpha)
    objects = np.random.choice(range(N), size=num_topics, p=probs, replace=False)

    # Choose the topics
    topics = [all_topics[i] for i in objects]

    # Pass the topics to the LLM
    output = send_request(topics)

    # Parse the JSON output
    json_parser.parse(output["choices"][0]["message"]["content"])




    # Generate a prompt
    prompt = f"Write a brief summary of the following topics: {', '.join(topics)}"

    # Generate a response
    response = client.complete(prompt, max_tokens=100)

    # Save the response
    with open(f"data/synthetic_data/{i}.txt", "w") as f:
        f.write(response.choices[0].text.strip())
