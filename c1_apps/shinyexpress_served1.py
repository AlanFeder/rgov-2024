import os
import sys

from shiny.express import input, ui

# Get the directory of the current script
current_dir = os.path.dirname(__file__)

# Move up to the parent directory and then to the cousin folder
cousin_folder = os.path.join(current_dir, "..", "b1_rag_fns")

# Add cousin folder to sys.path so it can be imported
sys.path.append(os.path.abspath(cousin_folder))

from b1_all_rag_fns import do_rag
from dotenv import load_dotenv

is_env = load_dotenv()

oai_api_key = os.getenv("OPENAI_API_KEY")
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

    # Create a chat instance and display it in the main panel
    chat = ui.Chat(id="chat")
    chat.ui()

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


# Define a callback to run when the user submits a message
@chat.on_user_submit
async def _():
    user_message = chat.user_input()
    response, _ = do_rag(
        user_input=user_message,
        n_results=3,
        stream=True,
        oai_api_key=oai_api_key,
        model_name=input.model_choice(),
    )
    # Append the response into the chat
    await chat.append_message_stream(response)
