import gradio as gr


import sys
import os

# Get the directory of the current script
current_dir = os.path.dirname(__file__)

# Move up to the parent directory and then to the cousin folder
cousin_folder = os.path.join(current_dir, "..", "b1_rag_fns")

# Add cousin folder to sys.path so it can be imported
sys.path.append(os.path.abspath(cousin_folder))

from dotenv import load_dotenv
from b1_all_rag_fns import do_rag


def gr_ch_if(user_input: str, history):
    oai_api_key = os.getenv("OPENAI_API_KEY")
    response, _ = do_rag(
        user_input, stream=False, n_results=3, model_name="gpt-4o-mini", oai_api_key=oai_api_key
    )
    return response


with gr.Blocks() as demo:

    gr.ChatInterface(
        fn=gr_ch_if,
        # type="messages",
        title="Use Gradio to Run RAG on the previous R/Gov Talks - Chat Interface 1",
    )

if __name__ == "__main__":
    demo.launch(share=True)
