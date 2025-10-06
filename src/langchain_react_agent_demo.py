#!/usr/bin/env python3

from demo_utils import getenv
api_key = getenv("GEMINI_API_KEY")
if not api_key:
    raise Exception("GEMINI_API_KEY needs to be set in .env.")

from demo_tools import ddg_search, get_tavily_search_tool, web_page_loader

from langchain.chat_models import init_chat_model
from langchain.agents import AgentExecutor, create_tool_calling_agent, create_react_agent

# from langchain_core.prompts import ChatPromptTemplate
# prompt = ChatPromptTemplate.from_messages(
#     [
#         (
#             "system",
#             "You are a helpful Pokémon GO event expert. You must use the provided tools to find "
#             "the answer to any question requiring current information. Once you have the information, "
#             "synthesize a concise and friendly answer for the user. "
#             "You have access to the following tools: "
#             " "
#             "{tools} "
#             " "
#         ),
#         ("placeholder", "{chat_history}"),
#         ("human", "{input}"),
#         ("placeholder", "{agent_scratchpad}"),
#     ]
# )

from langchain_core.prompts import PromptTemplate

template = '''Answer the following questions as best you can. You have access to the following tools:

{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}'''

prompt = PromptTemplate.from_template(template)


tools = [ddg_search, web_page_loader]
# tools = [get_tavily_search_tool(), web_page_loader]

model = init_chat_model("gemini-2.5-flash", model_provider="google_genai", api_key=api_key)
# agent = create_tool_calling_agent(model, tools, prompt)
agent = create_react_agent(model, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

#
#
#

question = "Find a list of potential pokemon go events that are active today. For each one find the details of the event. After verifying the Pokemon GO events that are active today and their details, give me a summary for each active event."
question = "Build a plan to find the details of the current pokemon go events. Share the plan, then execute the plan.  Show me the results."

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
