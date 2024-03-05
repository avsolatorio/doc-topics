import json
import requests
import fire
from bs4 import BeautifulSoup
from pathlib import Path
from tqdm.auto import tqdm

proj_dir = Path(__file__).parent.parent


def get_wb_topics():
    r = requests.get("https://www.worldbank.org/en/topic")
    soup = BeautifulSoup(r.content)

    topic_list = {}
    all_topics = []

    for topic in tqdm(soup.find_all(attrs={"class": "topic-card"}), desc="Scraping topics"):
        topic_name = topic.find("img").attrs["title"].strip()
        topic_list[topic_name] = [i.text.strip() for i in topic.find_all(attrs={"class": "topic-list-item"})]

        all_topics.append(topic_name)
        all_topics.extend(topic_list[topic_name])

    data = {"all_topics": all_topics, "topic_list": topic_list}

    with open(proj_dir / "data" / "wb_topics.json", "w") as f:
        json.dump(data, f, indent=2)

if __name__ == "__main__":
    # poetry run python scripts/get_wb_topics.py
    fire.Fire(get_wb_topics)
