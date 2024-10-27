import asyncio
import json

import yt_dlp as youtube_dl
from pyprojroot import here


async def download_audio_yt_dl(vid: dict, output_path: str):
    video_url = vid["VideoURL"]
    filename = f"vid_{vid['id0']}"
    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "128",
            }
        ],
        "outtmpl": f"{output_path}/{filename}.%(ext)s",
        "verbose": False,
    }

    try:
        await asyncio.to_thread(youtube_dl.YoutubeDL(ydl_opts).download, [video_url])
    except Exception as e:
        print(f"Failed to download {video_url}: {e}")


async def main():
    fp_data = here() / "data"

    try:
        with open(fp_data / "rgov_talks.json", "r") as f:
            dcr_data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading JSON file: {e}")
        return

    fp_audio = fp_data / "audio"
    fp_audio.mkdir(exist_ok=True)

    tasks = [download_audio_yt_dl(vid, str(fp_audio)) for vid in dcr_data]
    await asyncio.gather(*tasks)


if __name__ == "__main__":
    asyncio.run(main())
