import json

import whisper
from pyprojroot import here
from tqdm import tqdm

if __name__ == "__main__":
    fp_data = here() / "data"

    with open(fp_data / "rgov_talks.json", "r") as f:
        dcr_data = json.load(f)

    fp_audio = fp_data / "audio"

    model = whisper.load_model("base.en")

    for vid in tqdm(dcr_data):
        file_path = str(fp_audio / f"vid_{vid['id0']}.mp3")
        try:
            result = model.transcribe(file_path)
            vid["transcript"] = result["text"].strip()
        except Exception as e:
            print(f"{vid['id0']} failed with {e}")

    with open(fp_data / "rgov_talks_v2a.json", "w") as f:
        json.dump(dcr_data, f)
