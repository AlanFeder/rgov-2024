import os
import sys

import gradio as gr

# Get the directory of the current script
current_dir = os.path.dirname(__file__)

# Move up to the parent directory and then to the cousin folder
cousin_folder = os.path.join(current_dir, "..", "b1_rag_fns")

# Add cousin folder to sys.path so it can be imported
sys.path.append(os.path.abspath(cousin_folder))

from b1_all_rag_fns import do_rag
from dotenv import load_dotenv


def create_video_html(video_info: list) -> str:
    html = """
    <style>
        .video-container {
            display: flex;
            flex-wrap: wrap;
            justify-content: space-around;
        }
        .video-item {
            width: 30%;
            min-width: 300px;
            margin-bottom: 20px;
        }
        @media (max-width: 1200px) {
            .video-item {
                width: 45%;
            }
        }
        @media (max-width: 768px) {
            .video-item {
                width: 100%;
            }
        }
    </style>
    <div class="video-container">
    """

    for vid_info in video_info:
        yt_id = vid_info["VideoURL"].split("/")[-1].split("=")[-1]
        yt_url = f"https://www.youtube.com/embed/{yt_id}"
        html += f"""
        <div class="video-item">
            <h3>{vid_info["Title"]}</h3>
            <p><em>{vid_info["Speaker"]}</em></p>
            <p>Year: {vid_info["Year"]}</p>
            <p>Similarity Score: {100 * vid_info["score"]:.0f}/100</p>
            <iframe width="100%" height="215" src="{yt_url}" frameborder="0" allowfullscreen></iframe>
            <details>
                <summary>Transcript</summary>
                <p>{vid_info["transcript"]}</p>
            </details>
        </div>
        """
    html += "</div>"
    return html


def gr_ch_if(user_input: str, model_radio: str):
    model_name = "gpt-4o-mini" if model_radio == "Cheaper" else "gpt-4o"
    oai_api_key = os.getenv("OPENAI_API_KEY")
    response, retrieved_docs = do_rag(
        user_input,
        stream=False,
        n_results=3,
        oai_api_key=oai_api_key,
        model_name=model_name,
    )
    video_html = create_video_html(retrieved_docs)

    return response, video_html


# Create Gradio interface with single column layout
with gr.Blocks() as iface:
    gr.Markdown("# RAG on R/Gov Talks")
    gr.Markdown("Use Gradio to Run RAG on the previous R/Gov Talks")

    with gr.Row():
        with gr.Column(scale=1):
            model_radio = gr.Radio(
                choices=["Cheaper", "More Accurate"],
                value=0,
                label="Model",
                info="Choose the model to use",
                type="value",
                interactive=True,
            )

        with gr.Column(scale=3):
            query_input = gr.Textbox(label="Enter your question:")

    response_output = gr.Textbox(label="Response", interactive=False)
    video_output = gr.HTML(label="Relevant Videos")

    query_input.submit(
        fn=gr_ch_if,
        inputs=[query_input, model_radio],
        outputs=[response_output, video_output],
    )

    # Add the static markdown at the bottom
    gr.Markdown("""This Gradio app was created for Alan Feder's [talk at the 2024 R/Gov Conference](https://rstats.ai/gov.html). \n\n The Github repository that houses all the code is [here](https://github.com/AlanFeder/rgov-2024) -- feel free to fork it and use it on your own!"""
    )
    gr.Markdown("***")
    gr.Markdown("### Contact me!")
    gr.Image("https://raw.githubusercontent.com/AlanFeder/rgov-2024/refs/heads/main/AJF_Headshot.jpg", width=60)
    gr.Markdown(
        """
        [Email](mailto:AlanFeder@gmail.com) | [Website](https://www.alanfeder.com/) | [LinkedIn](https://www.linkedin.com/in/alanfeder/) | [GitHub](https://github.com/AlanFeder)
        """
    )


# Launch the app
if __name__ == "__main__":
    iface.launch(
        share=True,
        favicon_path="favicon_io/favicon.ico",
    )
