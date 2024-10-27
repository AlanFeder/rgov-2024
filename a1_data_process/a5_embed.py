from pyprojroot import here
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
# from pathlib import Path
# import shutil
import numpy as np


if __name__ == "__main__":
    load_dotenv()
    oai_api_key = os.getenv("OPENAI_API_KEY")
    oai_client = OpenAI(api_key=oai_api_key)

    fp_data = here() / "data"
    fp_audio = fp_data / "audio"

    with open(fp_data / "rgov_talks_v3.json", "r") as f:
        dcr_data = json.load(f)

    all_abstracts = [vid["Abstract"] for vid in dcr_data]
    all_embeds_responses = oai_client.embeddings.create(
        input=all_abstracts, model="text-embedding-3-small"
    )
    all_embeds = np.stack([ee.embedding for ee in all_embeds_responses.data])

    np.savetxt(
        fp_data / "embeds.csv",
        all_embeds,
        delimiter=",",
        fmt="%0.16f",
    )
