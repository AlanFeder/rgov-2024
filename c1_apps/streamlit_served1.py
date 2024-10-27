import os

import streamlit as st
from dotenv import load_dotenv

from b1_rag_fns.b1_all_rag_fns import do_rag


def run_app():
    st.set_page_config(
        page_title="Streamlit RAG on R/Gov Talks",
        page_icon="favicon_io/favicon.ico",
        layout="wide",
        initial_sidebar_state="auto",
        menu_items=None,
    )
    st.title("Use Streamlit to Run RAG on the previous R/Gov Talks")

    load_dotenv()

    oai_api_key = os.getenv("OPENAI_API_KEY")

    with st.sidebar:
        model_name = st.radio(
            label="Which GPT model\ndo you want to use?",
            options=["gpt-4o-mini", "gpt-4o"],
            format_func=lambda x: "Cheaper" if x == "gpt-4o-mini" else "More Accurate",
            key="model_name",
            index=0,
            horizontal=False,
        )

    # How many results to bring to the generator
    n_results = 3

    # Get user input
    user_input = st.text_input("Enter your question:")
    if user_input:
        response, retrieved_docs = do_rag(
            user_input=user_input,
            oai_api_key=oai_api_key,
            stream=True,
            n_results=n_results,
            model_name=model_name,
        )

        # Display the response
        st.write_stream(response)

        st.divider()
        st.subheader("RAG-identified relevant videos")
        n_vids = len(retrieved_docs)
        if n_vids == 0:
            st.markdown("No relevant videos identified")
        elif n_vids == 1:
            _, vid_c1, _ = st.columns(3)
            vid_containers = [vid_c1]
        elif n_vids == 2:
            _, vid_c1, vid_c2, _ = st.columns([1 / 6, 1 / 3, 1 / 3, 1 / 6])
            vid_containers = [vid_c1, vid_c2]
        elif n_vids > 2:
            vid_containers = st.columns(n_vids)
        for i, vid_info in enumerate(retrieved_docs):
            vid_container = vid_containers[i]
            with vid_container:
                vid_title = vid_info["Title"]
                vid_speaker = vid_info["Speaker"]
                sim_score = 100 * vid_info["score"]
                vid_url = vid_info["VideoURL"]
                st.markdown(
                    f"**{vid_title}**\n\n*{vid_speaker}*\n\nYear: {vid_info['Year']}"
                )
                st.caption(f"Similarity Score: {sim_score:.0f}/100")
                st.video(vid_url)
                with st.expander(label="Transcript", expanded=False):
                    st.markdown(vid_info["transcript"])

        st.divider()

        st.caption(
            """This streamlit app was created for Alan Feder's [talk at the 2024 R/Gov Conference](https://rstats.ai/nyr.html). . \n\n The Github repository that houses all the code is [here](https://github.com/AlanFeder/rgov-2024) -- feel free to fork it and use it on your own!"""
        )
        # \n\n The slides used are [here](https://bit.ly/nyr-rag)
        st.divider()

        st.subheader("Contact me!")
        st.image("AJF_Headshot.jpg", width=60)
        st.markdown(
            "[Email](mailto:AlanFeder@gmail.com) | [Website](https://www.alanfeder.com/) | [LinkedIn](https://www.linkedin.com/in/alanfeder/) | [GitHub](https://github.com/AlanFeder)"
        )


if __name__ == "__main__":
    run_app()
