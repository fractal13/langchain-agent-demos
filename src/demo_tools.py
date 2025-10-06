#!/usr/bin/env python3

from langchain_core.tools import tool
from ddgs import DDGS

@tool
def ddg_search(query: str) -> str:
    """
    Performs a DuckDuckGo search for up-to-date information.
    Use this tool exclusively for answering current event or news-related questions.
    Returns a formatted summary of the top search results (max 3 snippets).
    """
    print(f"\n--- Agent is executing tool: ddg_search with query: '{query}' ---")

    try:
        # Use DDGS().text to get up to 3 text-based search results
        search_results = DDGS().text(query, max_results=3)

        if not search_results:
            return "No current DuckDuckGo search results found."

        # Format the results into a single string for the LLM
        formatted_results = "Current Search Results Summary:\n"
        for i, result in enumerate(search_results):
            formatted_results += (
                f"{i + 1}. Title: {result.get('title', 'N/A')}. "
                f"Snippet: {result.get('body', 'N/A')}\n"
            )

        print("--- Tool execution successful. Returning results to LLM. ---\n")
        return formatted_results

    except Exception as e:
        print(f"--- DDGS Error: {e} ---")
        return f"Error occurred during search: {e}"


from demo_utils import getenv
TAVILY_API_KEY=getenv("TAVILY_API_KEY")

from langchain_tavily import TavilySearch

def get_tavily_search_tool():
    tool = TavilySearch(
        max_results=3,
        topic="general",
    )
    return tool

def main():
    query = "Who is Walt Disney?"
    answer = ddg_search.invoke(query)
    print(f"ddgs: Q: {query} --> A: {answer}")

    tavily = get_tavily_search_tool()
    answer = tavily.invoke(query)
    print(f"tavily: Q: {query} --> A: {answer}")
    return

if __name__ == "__main__":
    main()

