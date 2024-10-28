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


def gr_ch_if(user_input: str, history):
    oai_api_key = os.getenv("OPENAI_API_KEY")
    response, _ = do_rag(
        user_input,
        stream=False,
        n_results=3,
        model_name="gpt-4o-mini",
        oai_api_key=oai_api_key,
    )
    return response


with gr.Blocks() as demo:
    gr.ChatInterface(
        fn=gr_ch_if,
        # type="messages",
        title="Use Gradio to Run RAG on the previous R/Gov Talks - Chat Interface 1",
    )

    # Add the static markdown at the bottom
    gr.Markdown(
        """
        This Gradio app was created for Alan Feder's [talk at the 2024 R/Gov Conference](https://rstats.ai/gov.html). \n\n The Github repository that houses all the code is [here](https://github.com/AlanFeder/rgov-2024) -- feel free to fork it and use it on your own!
        """
    )
    gr.Divider()
    gr.Subheader("Contact me!")
    gr.Image("AJF_Headshot.jpg", width=60)
    gr.Markdown(
        """
        [Email](mailto:AlanFeder@gmail.com) | [Website](https://www.alanfeder.com/) | [LinkedIn](https://www.linkedin.com/in/alanfeder/) | [GitHub](https://github.com/AlanFeder)
        """
    )

if __name__ == "__main__":
    demo.launch(
        share=True,
        favicon_path="https://raw.githubusercontent.com/AlanFeder/rgov-2024/refs/heads/main/favicon_io/favicon.ico",
    )
