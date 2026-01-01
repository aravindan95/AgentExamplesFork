import os
from dotenv import load_dotenv
from datetime import date
from tavily import TavilyClient
import json

# LangGraph and LangChain imports
from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_core.tools import Tool

# Prompt components
from prompts import role, goal, instructions, knowledge

# Load environment variables
load_dotenv()

# Initialize Tavily client
tavily_api_key = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=tavily_api_key)


# Define the state for the graph
class State(TypedDict):
    messages: Annotated[list, add_messages]


class Agent:
    def __init__(self, model="gpt-4o-mini"):
        """
        Initialize the LangGraph agent using create_react_agent.

        Args:
            model (str): The language model to use
        """
        self.name = "LangGraph Agent"
        # Create tools
        self.tools = self._create_tools()

        # Create memory
        self.memory = MemorySaver()
        # Memory will be checkpointed per thread. We will start with thread id 1.
        self.thread_id = 1

        # Create the system prompt
        self.system_prompt = "\n".join([role, goal, instructions, knowledge])

        # Initialize the language model
        self.llm = ChatOpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=model,
            temperature=0
        )

        # Create the agent graph
        self.graph = create_agent(
            model=self.llm,
            tools=self.tools,
            system_prompt=self.system_prompt,
            checkpointer=self.memory
        )


    @staticmethod
    def date_tool(tool_input={}):
        """
        Function to get the current date.
        """
        today = date.today()
        return today.strftime("%B %d, %Y")

    @staticmethod
    def web_search(query):
        """
        This function searches the web for the given query and returns the results.
        """
        # Call Tavily's search and dump the results as a JSON string
        search_response = tavily_client.search(query)
        results = json.dumps(search_response.get('results', []))
        print(f"Web Search Results for '{query}':")
        print(results)
        return results

    def _create_tools(self):
        """
        Create tools for the agent.

        Returns:
            List of tools
        """
        return [
            Tool(
                name="date",
                func=self.date_tool,
                description="Useful for getting the current date"
            ),
            Tool(
                name="web_search",
                func=self.web_search,
                description="Useful for searching the web for information"
            )
        ]

    def _inc_thread_id(self):
        """
        Simply increments the thread id and returns the new id.

        """
        new_thread_id = self.thread_id + 1
        self.thread_id = new_thread_id
        return new_thread_id

    def chat(self, message):
        """
        Send a message and get a response.

        Args:
            message (str): User's input message

        Returns:
            str: Assistant's response
        """
        try:
            # Prepare input
            inputs = {"messages": [("user", message)]}
            config = {"configurable": {"thread_id": str(self.thread_id)}}

            # Stream the graph updates and collect the final response
            full_response = ""
            for event in self.graph.stream(inputs, config=config, stream_mode="values"):
                if event and "messages" in event:
                    last_message = event["messages"][-1]
                    if hasattr(last_message, "content"):
                        full_response = last_message.content

            return full_response

        except Exception as e:
            print(f"Error in chat: {e}")
            return "Sorry, I encountered an error processing your request."

    def clear_chat(self):
        """
        Reset the conversation context.

        Returns:
            bool: True if reset was successful
        """
        try:
            self._inc_thread_id() # Incrementing the thread ID basically resets the memory
            return True
        except Exception as e:
            print(f"Error clearing chat: {e}")
            return False


def main():
    """
    Example usage demonstrating the agent interface.
    """
    agent = Agent()

    while True:
        query = input("You: ")
        if query.lower() in ['exit', 'quit']:
            break

        response = agent.chat(query)
        print(f"Assistant: {response}")


if __name__ == "__main__":
    main()
