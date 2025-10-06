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


from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

@tool
def web_page_loader(url: str) -> str:
    """
    Loads and extracts the primary content from a specific URL for deep analysis.
    Use this tool when you have a specific, direct URL (must start with 'http' or 'https').
    Returns a summarized portion (first 3000 characters) of the page content.
    """
    print(f"\n--- Agent is executing tool: web_page_loader with URL: '{url}' ---")
    try:
        # 1. Load the document using WebBaseLoader
        loader = WebBaseLoader(url)
        # Using .load() for synchronous loading
        docs = loader.load()

        if not docs:
            return f"Could not load any content from the URL: {url}"

        # 2. Split the document to manage token limits (Crucial for large web pages)
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=50,
            # Use common separators to maintain semantic coherence
            separators=["\n\n", "\n", " ", ""]
        )
        split_docs = text_splitter.split_documents(docs)

        # 3. Join the first few chunks to provide a concise summary of the content
        content = " ".join([doc.page_content for doc in split_docs[:3]])

        # 4. Truncate the content safely for the LLM's context window
        max_return_length = 3000
        summary = content[:max_return_length]
        if len(content) > max_return_length:
            summary += f"... [Content truncated. Total chunks available: {len(split_docs)}]"

        print("--- Tool execution successful. Returning page content summary to LLM. ---\n")
        return f"Content loaded from {url}:\n\n{summary}"

    except Exception as e:
        # Catch common network or parsing errors
        print(f"--- WebLoader Error: {e} ---")
        return f"Error occurred during web page loading: {e}. Ensure the URL is accessible and correct."

def main():
    query = "Who is Walt Disney?"
    answer = ddg_search.invoke(query)
    print(f"ddgs: Q: {query} --> A: {answer}")

    tavily = get_tavily_search_tool()
    answer = tavily.invoke(query)
    print(f"tavily: Q: {query} --> A: {answer}")

    url = "https://cs.utahtech.edu/cs/3005/assignments/assignment_09_color_table/"
    content = web_page_loader.invoke(url)
    print(f"web page loader: Y: {url} --> C: {content}")


    return

if __name__ == "__main__":
    main()

