#!/usr/bin/env python3

from demo_utils import getenv
api_key = getenv("GEMINI_API_KEY")
if not api_key:
    raise Exception("GEMINI_API_KEY needs to be set in .env.")

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

model = init_chat_model("gemini-2.5-flash", model_provider="google_genai", api_key=api_key)

messages = [
    SystemMessage(content="Translate the following from English into Italian"),
    HumanMessage(content="hi!"),
]

result = model.invoke(messages)
print(result.content)


