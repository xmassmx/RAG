"""A Simple chatbot that uses the LangChain and Gradio UI to answer questions about wandb documentation."""
import os
from types import SimpleNamespace
from dotenv import load_dotenv

import gradio as gr
import wandb
from chain import get_answer, load_chain, load_vector_store
from config import default_config
load_dotenv()
os.environ["LANGCHAIN_WANDB_TRACING"] = "true"

# wandb documentation to configure wandb using env variables
# https://docs.wandb.ai/guides/track/advanced/environment-variables
# here we are configuring the wandb project name
os.environ["WANDB_PROJECT"] = default_config.project


class Chat:
    """A chatbot interface that persists the vectorstore and chain between calls."""

    def __init__(
        self,
        config: SimpleNamespace,
    ):
        """Initialize the chatbot.
        Args:
            config (SimpleNamespace): The configuration.
        """
        self.config = config
        self.wandb_run = wandb.init(
            project=self.config.project,
            entity=self.config.entity,
            job_type=self.config.job_type,
            config=self.config,
        )
        self.vector_store = None
        self.chain = None

    def __call__(
        self,
        question: str,
        history: list[tuple[str, str]] or None = None,
        openai_api_key: str = None,
    ):
        """Answer a question about wandb documentation using the LangChain QA chain and vector store retriever.
        Args:
            question (str): The question to answer.
            history (list[tuple[str, str]] | None, optional): The chat history. Defaults to None.
            openai_api_key (str, optional): The OpenAI API key. Defaults to None.
        Returns:
            list[tuple[str, str]], list[tuple[str, str]]: The chat history before and after the question is answered.
        """
        if openai_api_key is not None:
            openai_key = openai_api_key
        elif os.environ["OPENAI_API_KEY"]:
            openai_key = os.environ["OPENAI_API_KEY"]
        else:
            raise ValueError(
                "Please provide your OpenAI API key as an argument or set the OPENAI_API_KEY environment variable"
            )
        if self.vector_store is None:
            self.vector_store = load_vector_store(
                wandb_run=self.wandb_run, openai_api_key=openai_key
            )
        if self.chain is None:
            self.chain = load_chain(
                self.wandb_run, self.vector_store, openai_api_key=openai_key
            )

        history = history or []
        question = question.lower()
        response = get_answer(
            chain=self.chain,
            question=question,
            chat_history=history,
        )
        history.append((question, response))
        return history, history


with gr.Blocks() as demo:
    gr.HTML(
        """<div style="text-align: center; max-width: 700px; margin: 0 auto;">
        <div
        style="
            display: inline-flex;
            align-items: center;
            gap: 0.8rem;
            font-size: 1.75rem;
        "
        >
        <h1 style="font-weight: 900; margin-bottom: 7px; margin-top: 5px;">
            Retrieval-Augmented Generation Bot
        </h1>
        </div>
        <p style="margin-bottom: 10px; font-size: 94%">
        Hi, I'm a Q and A bot which can address questions related to the project/assignment reports of Muhammad Abdullah Mulkana. Start by typing in your OpenAI API key, then ask your questions/issues and then press enter.<br>
        Built using <a href="https://langchain.readthedocs.io/en/latest/" target="_blank">LangChain</a> and <a href="https://github.com/gradio-app/gradio" target="_blank">Gradio Github repo</a>
        </p>
    </div>"""
    )
    with gr.Row():
        question = gr.Textbox(
            label="Type in your questions here and press Enter!",
            placeholder="what are the documents about?",
        )
        openai_api_key = gr.Textbox(
            type="password",
            label="Enter your OpenAI API key here",
        )
    state = gr.State()
    chatbot = gr.Chatbot()
    question.submit(
        Chat(
            config=default_config,
        ),
        [question, state, openai_api_key],
        [chatbot, state],
    )


if __name__ == "__main__":
    demo.queue().launch(
        share=False, server_name="0.0.0.0", server_port=8884, show_error=True
    )