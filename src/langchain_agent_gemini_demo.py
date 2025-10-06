#!/usr/bin/env python3

from demo_utils import getenv
api_key = getenv("GEMINI_API_KEY")
if not api_key:
    raise Exception("GEMINI_API_KEY needs to be set in .env.")

from demo_tools import ddg_search, get_tavily_search_tool

from langchain.chat_models import init_chat_model
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful Pokémon GO event expert. You must use the provided tools to find "
            "the answer to any question requiring current information. Once you have the information, "
            "synthesize a concise and friendly answer for the user."
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)
tools = [ddg_search]
# tools = [get_tavily_search_tool()]

model = init_chat_model("gemini-2.5-flash", model_provider="google_genai", api_key=api_key)
agent = create_tool_calling_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

#
#
#

question = "What events are happening in Pokemon GO today?"

print(f"--- Running Agent with Query: '{question}' ---\n")

try:
    result = agent_executor.invoke({"input": question})

    print("\n==================================================")
    print("             AGENT FINAL ANSWER")
    print("==================================================")
    print(result["output"])
    print("==================================================")

except Exception as e:
    print(f"\nAn error occurred during agent execution: {e}")
    print("Please ensure your GEMINI_API_KEY is correct and the LangChain packages are installed.")


if __name__ == '__main__':
    # Add a main execution block if needed, but the script runs top-to-bottom.
    pass
