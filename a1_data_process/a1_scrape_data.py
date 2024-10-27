import json
from collections import defaultdict

import requests
from bs4 import BeautifulSoup
from pyprojroot import here

base_url = "https://rstats.ai/videos"

response = requests.get(base_url)
soup = BeautifulSoup(response.content, "lxml")

script_tag = soup.find(
    lambda tag: tag.name == "script" and tag.text.strip().startswith('{"x":')
)
my_data = json.loads(script_tag.text)["x"]["tag"]["attribs"]["data"]
my_data = [dict(zip(my_data.keys(), values)) for values in zip(*my_data.values())]
dcr_data = []
# Create a defaultdict to keep track of counts for each year
year_counter = defaultdict(int)

# List of keys you want to keep
keys_to_keep = ["Year", "Speaker", "Title", "Abstract", "VideoURL"]
for vid in my_data:
    if "New York" not in vid["Conference"]:
        if vid["Type"] != "Highlights":
            if vid["VideoURL"]:
                my_vid = {key: vid[key] for key in keys_to_keep if key in vid}
                year = str(vid["Year"])  # Ensure the Year is a string
                year_counter[year] += 1  # Increment the count for the current year
                my_vid["id0"] = f"{year}_{year_counter[year]:02d}"  # Format the id0
                dcr_data.append(my_vid)


with open(here() / "data" / "rgov_talks.json", "w") as f:
    json.dump(dcr_data, f)
