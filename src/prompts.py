"""Prompts for the chatbot and evaluation."""
import json
import logging
import pathlib
from typing import Union

from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

logger = logging.getLogger(__name__)


def load_chat_prompt(f_name: Union[pathlib.Path, str] = None) -> ChatPromptTemplate:
    if isinstance(f_name, str) and f_name:
        f_name = pathlib.Path(f_name)
    if f_name and f_name.is_file():
        template = json.load(f_name.open("r"))
    else:
        logger.warning(
            f"No chat prompt provided. Using default chat prompt from {__name__}"
        )
        template = {
            "system_template": "You are wandbot, an AI assistant designed to provide accurate and helpful responses to questions related to the reports of my university projects.\\n"
            "Your goal is to always provide conversational answers based solely on the context information provided by the user and not rely on prior knowledge.\\n\\n"
            "If you are unable to answer a question or generate valid code or links based on the context provided, respond with 'Hmm, I'm not sure'.\\n\\n"
            "You can only answer questions related to these few assignments and projects.\\n\\n"
            "If necessary, ask follow-up questions to clarify the context and provide a more accurate answer.\\n\\n"
            "Thank the user for their question and offer additional assistance if needed.\\nALWAYS prioritize accuracy and helpfulness in your responses.\\n\\n"
            "CONTEXT\\n{context}\\n================\\nGiven the context information and not prior knowledge, answer the question.\\n================\\n", 
            "human_template": "{question}\\n================\\nFinal Answer:"}

    messages = [
        SystemMessagePromptTemplate.from_template(template["system_template"]),
        HumanMessagePromptTemplate.from_template(template["human_template"]),
    ]
    prompt = ChatPromptTemplate.from_messages(messages)
    return prompt


def load_eval_prompt(f_name: Union[pathlib.Path, str] = None) -> ChatPromptTemplate:
    if isinstance(f_name, str) and f_name:
        f_name = pathlib.Path(f_name)
    if f_name and f_name.is_file():
        human_template = f_name.open("r").read()
    else:
        logger.warning(
            f"No human prompt provided. Using default human prompt from {__name__}"
        )

        human_template = """\nQUESTION: {query}\nCHATBOT ANSWER: {result}\n
        ORIGINAL ANSWER: {answer} GRADE:"""

    system_message_prompt = SystemMessagePromptTemplate.from_template(
        """You are an evaluator for the W&B chatbot.You are given a question, the chatbot's answer, and the original answer, 
        and are asked to score the chatbot's answer as either CORRECT or INCORRECT. Note 
        that sometimes, the original answer is not the best answer, and sometimes the chatbot's answer is not the 
        best answer. You are evaluating the chatbot's answer only. Example Format:\nQUESTION: question here\nCHATBOT 
        ANSWER: student's answer here\nORIGINAL ANSWER: original answer here\nGRADE: CORRECT or INCORRECT here\nPlease 
        remember to grade them based on being factually accurate. Begin!"""
    )
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [system_message_prompt, human_message_prompt]
    )
    return chat_prompt
