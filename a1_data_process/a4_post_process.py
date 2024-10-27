from pyprojroot import here
from openai import OpenAI
from dotenv import load_dotenv
import os
import json
# from pathlib import Path
# import shutil

SYSTEM_PROMPT = """
You are a helpful assistant that inputs the transcribed test of a talk and outputs a short abstract of the talk.  This abstract will be used for a Retrieval/RAG System.  I will provide comparable abstracts below to guide the length.  Your output should use first person when relevant
---
[EXAMPLE 1]
Do you ever find yourself starting with a simple analysis script only to end up wrangling a thousand line behemoth? Are you sick of wasting time re-running long scripts from start to finish, just to make sure everything is up-to-date? Are you haphazardly saving objects to file because they take a long time to generate? There's got to be a better way! Enter targets, an R package used to build reproducible, efficient, and scalable pipelines. In this talk, I'll introduce the targets package and share how I've used it to streamline my work modelling infectious disease spread at the Public Health Agency of Canada.
---
[EXAMPLE 2]
Simulations are often run to benchmark a method using data where the results are known or to compare a few methods on a nicely structured (simulated) data. Simulating data in R is not hard. If you have to simulate many different datasets, tweaking some parameters, how to automate such a process to run multiple times and maximize the use of computer or server resources.  In this talk I will show how I was able to run multiple simulations at the same time using the doParallel package to run a few R threads simultaneously (from within R) to simulated multiple datasets with genetics data under different scenarios.
"""


def make_abstract(transcript: str, oai_client: OpenAI) -> str:
    completion = oai_client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": transcript},
        ],
        temperature=0.8,
        seed=18,
    )
    output = completion.choices[0].message.content
    return output


if __name__ == "__main__":
    load_dotenv()
    oai_api_key = os.getenv("OPENAI_API_KEY")
    oai_client = OpenAI(api_key=oai_api_key)

    fp_data = here() / "data"
    fp_audio = fp_data / "audio"

    with open(fp_data / "rgov_talks_v2a.json", "r") as f:
        dcr_data = json.load(f)

    for vid in dcr_data:
        if not vid["Abstract"]:
            print(vid['id0'])
            vid["Abstract"] = make_abstract(vid["transcript"], oai_client)

    with open(fp_data / "rgov_talks_v3.json", "w") as f:
        json.dump(dcr_data, f)

    # shutil.rmtree(fp_audio)
