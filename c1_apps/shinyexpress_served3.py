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

# Enhanced CSS with animations and better styling
ui.tags.style(
    """
    :root {
        --primary-color: #2c3e50;
        --secondary-color: #34495e;
        --accent-color: #3498db;
        --bg-light: #f8f9fa;
        --shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        --transition: all 0.3s ease;
    }

    body {
        background-color: #f5f6fa;
    }

    .main-title {
        color: var(--primary-color);
        margin-bottom: 30px;
        font-weight: 600;
        border-bottom: 3px solid var(--accent-color);
        padding-bottom: 10px;
        transition: var(--transition);
    }

    .main-title:hover {
        color: var(--accent-color);
    }

    .card {
        border: none !important;
        box-shadow: var(--shadow) !important;
        transition: var(--transition) !important;
        margin-bottom: 25px !important;
        border-radius: 12px !important;
        overflow: hidden;
    }

    .card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15) !important;
    }

    .card-header {
        background-color: var(--primary-color) !important;
        color: white !important;
        font-weight: 500;
        padding: 15px 20px !important;
        border-bottom: none !important;
    }

    .contact-section {
        background: linear-gradient(145deg, var(--bg-light), #ffffff);
        padding: 25px;
        border-radius: 15px;
        margin-top: 30px;
        box-shadow: var(--shadow);
        transition: var(--transition);
    }

    .contact-section:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }

    .contact-section img {
        border-radius: 50%;
        margin-right: 20px;
        vertical-align: middle;
        border: 3px solid var(--accent-color);
        transition: var(--transition);
    }

    .contact-section img:hover {
        transform: scale(1.1);
    }

    .contact-links {
        display: inline-block;
        vertical-align: middle;
    }

    .contact-links a {
        color: var(--accent-color);
        text-decoration: none;
        margin: 0 10px;
        transition: var(--transition);
    }

    .contact-links a:hover {
        color: var(--primary-color);
        text-decoration: underline;
    }

    .btn-primary {
        background-color: var(--accent-color) !important;
        border: none !important;
        padding: 10px 25px !important;
        transition: var(--transition) !important;
    }

    .btn-primary:hover {
        background-color: var(--primary-color) !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }

    .video-container {
        display: flex;
        flex-wrap: wrap;
        gap: 25px;
        justify-content: space-around;
        padding: 20px;
    }

    .video-item {
        width: 30%;
        min-width: 300px;
        background: #ffffff;
        border-radius: 12px;
        padding: 20px;
        box-shadow: var(--shadow);
        transition: var(--transition);
    }

    .video-item:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }

    .video-item h3 {
        color: var(--primary-color);
        margin-bottom: 15px;
        font-weight: 600;
    }

    .video-item p {
        color: var(--secondary-color);
        margin: 8px 0;
        line-height: 1.6;
    }

    .video-item iframe {
        border-radius: 10px;
        margin: 15px 0;
        width: 100%;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }

    .video-item details {
        margin-top: 15px;
        border-top: 1px solid #eee;
        padding-top: 15px;
    }

    .video-item details summary {
        cursor: pointer;
        color: var(--accent-color);
        font-weight: 500;
        transition: var(--transition);
    }

    .video-item details summary:hover {
        color: var(--primary-color);
    }

    /* Responsive adjustments */
    @media (max-width: 1200px) {
        .video-item {
            width: 45%;
        }
    }

    @media (max-width: 768px) {
        .video-item {
            width: 100%;
        }
        .contact-section {
            text-align: center;
        }
        .contact-section img {
            margin-bottom: 15px;
        }
    }

    /* Input styling */
    .form-control {
        border-radius: 8px !important;
        border: 1px solid #dee2e6 !important;
        padding: 12px !important;
        transition: var(--transition) !important;
    }

    .form-control:focus {
        border-color: var(--accent-color) !important;
        box-shadow: 0 0 0 0.2rem rgba(52, 152, 219, 0.25) !important;
    }
"""
)

ui.page_opts(
    title="Use Shiny to Run RAG on the previous R/Gov Talks",
    fillable=True,
    fillable_mobile=True,
)

ui.h2("RAG on R/Gov Talks", class_="main-title")

with ui.layout_sidebar():
    with ui.sidebar():
        ui.div(
            {"style": "padding: 20px;"},
            ui.h3("Settings", style="color: #2c3e50; margin-bottom: 20px;"),
            ui.input_password(
                "api_key",
                "OpenAI API Key:",
                placeholder="Enter your API key",
                width="100%",
            ),
            ui.hr(),
            ui.input_radio_buttons(
                "model_choice",
                "Select Model:",
                choices={"gpt-4o-mini": "Cheaper", "gpt-4o": "More Accurate"},
                selected="gpt-4o-mini",
            ),
            ui.hr(),
            # About section moved to sidebar
            ui.h3(
                "About", style="color: #2c3e50; margin-top: 20px; margin-bottom: 10px;"
            ),
            ui.markdown(
                """
                This app was created for Alan Feder's [talk at the 2024 R/Gov Conference](https://rstats.ai/gov.html).

                The Github repository that houses all the code is [here](https://github.com/AlanFeder/rgov-2024) -- feel free to fork it and use it on your own!
                """
            ),
            ui.hr(),
            # Contact section moved to sidebar
            ui.h3(
                "Contact me!",
                style="color: #2c3e50; margin-top: 20px; margin-bottom: 10px;",
            ),
            ui.div(
                {"class": "contact-section", "style": "padding: 10px 0;"},
                ui.img(
                    src="https://raw.githubusercontent.com/AlanFeder/rgov-2024/refs/heads/main/AJF_Headshot.jpg",
                    width="60px",
                ),
                ui.div(
                    {"class": "contact-links"},
                    ui.markdown(
                        """
                        [Email](mailto:AlanFeder@gmail.com) | [Website](https://www.alanfeder.com/) | [LinkedIn](https://www.linkedin.com/in/alanfeder/) | [GitHub](https://github.com/AlanFeder)
                        """
                    ),
                ),
            ),
        )

    with ui.card(class_="query-card"):
        ui.card_header("Ask a Question")
        ui.input_text(
            id="query1",
            label="What question do you want to ask?",
            placeholder="What is the tidyverse?",
            width="100%",
        )
        ui.div(
            {"style": "padding: 15px 0;"},
            ui.input_action_button(
                "run_rag", "Submit!", class_="btn-primary", width="200px"
            ),
        )

    with ui.card(class_="response-card"):
        ui.card_header("AI Response")

        @render.ui
        def render_response():
            ans = rag_answer()
            return ui.markdown(ans)

    with ui.card():
        ui.card_header("Related Videos")

        @render.ui
        def create_video_html():
            video_info = list_retrieved_docs()
            html = '<div class="video-container">'

            for vid_info in video_info:
                yt_id = vid_info["VideoURL"].split("/")[-1].split("=")[-1]
                yt_url = f"https://www.youtube.com/embed/{yt_id}"
                html += f"""
                <div class="video-item">
                    <h3>{vid_info["Title"]}</h3>
                    <p><em>{vid_info["Speaker"]}</em></p>
                    <p>Year: {vid_info["Year"]}</p>
                    <p>Similarity Score: {100 * vid_info["score"]:.0f}/100</p>
                    <iframe height="215" src="{yt_url}" frameborder="0" allowfullscreen></iframe>
                    <details>
                        <summary>Transcript</summary>
                        <p>{vid_info["transcript"]}</p>
                    </details>
                </div>
                """
            html += "</div>"
            return ui.HTML(html)


rag_answer = reactive.value("")
list_retrieved_docs = reactive.value([])


@reactive.effect
@reactive.event(input.run_rag)
def do_rag_shiny():
    oai_api_key = input.api_key()
    response, retrieved_docs = do_rag(
        user_input=input.query1(),
        n_results=3,
        stream=False,
        oai_api_key=oai_api_key,
        model_name=input.model_choice(),
    )

    rag_answer.set(response)
    list_retrieved_docs.set(retrieved_docs)
