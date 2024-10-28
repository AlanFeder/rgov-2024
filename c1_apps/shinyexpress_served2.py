import os
import sys

from shiny import reactive
from shiny.express import input, render, ui

# Get the directory of the current script
current_dir = os.path.dirname(__file__)

# Move up to the parent directory and then to the cousin folder
cousin_folder = os.path.join(current_dir, "..", "b1_rag_fns")

# Add cousin folder to sys.path so it can be imported
sys.path.append(os.path.abspath(cousin_folder))

from b1_all_rag_fns import do_rag
from dotenv import load_dotenv

is_env = load_dotenv()


ui.page_opts(
    title="Use Shiny to Run RAG on the previous R/Gov Talks",
    fillable=True,
    fillable_mobile=True,
)


with ui.layout_sidebar():
    # Add radio buttons in the sidebar
    with ui.sidebar():
        ui.input_radio_buttons(
            "model_choice",
            "Select Model:",
            choices={"gpt-4o-mini": "Cheaper", "gpt-4o": "More Accurate"},
            selected="gpt-4o-mini",
        )

    ui.input_text(
        id="query1",
        label="What question do you want to ask?",
        placeholder="What is the tidyverse?",
    )
    ui.input_action_button("run_rag", "Submit!")


rag_answer = reactive.value("")
list_retrieved_docs = reactive.value([])


@reactive.effect
@reactive.event(input.run_rag)
def do_rag_shiny():
    oai_api_key = os.getenv("OPENAI_API_KEY")
    response, retrieved_docs = do_rag(
        user_input=input.query1(),
        n_results=3,
        stream=False,
        oai_api_key=oai_api_key,
        model_name=input.model_choice(),
    )

    rag_answer.set(response)
    list_retrieved_docs.set(retrieved_docs)


@render.ui
def render_response():
    ans = rag_answer()
    return ui.markdown(ans)


@render.ui
def create_video_html():
    video_info = list_retrieved_docs()
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
    return ui.HTML(html)


ui.markdown(
    """
    This app was created for Alan Feder's [talk at the 2024 R/Gov Conference](https://rstats.ai/gov.html).

    The Github repository that houses all the code is [here](https://github.com/AlanFeder/rgov-2024) -- feel free to fork it and use it on your own!
    """
)
ui.hr()  # Divider
ui.h3("Contact me!")
ui.img(
    src="https://raw.githubusercontent.com/AlanFeder/rgov-2024/refs/heads/main/AJF_Headshot.jpg",
    width="60px",
)
ui.markdown(
    """
    [Email](mailto:AlanFeder@gmail.com) | [Website](https://www.alanfeder.com/) | [LinkedIn](https://www.linkedin.com/in/alanfeder/) | [GitHub](https://github.com/AlanFeder)
    """
)
