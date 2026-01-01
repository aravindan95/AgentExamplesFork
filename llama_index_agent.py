import os
import asyncio
from dotenv import load_dotenv
from datetime import date
from tavily import TavilyClient
import json

# Llama-Index imports
from llama_index.llms.openai import OpenAI
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.core.memory import ChatMemoryBuffer

from prompts import role, goal, instructions, knowledge, llama_index_react_prompt

# Load environment variables
load_dotenv()

# Initialize Tavily client
tavily_api_key = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=tavily_api_key)


class Agent:
    def __init__(self, model="gpt-4o-mini"):
        """
        Initialize the Llama-Index agent.

        Args:
            model (str): The language model to use
        """
        self.name = "Llama-Index Agent"
        # Initialize the language model
        self.llm = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            model=model
        )

        # Create tools
        self.tools = self._create_tools()

        # Initialize the memory
        self.memory = ChatMemoryBuffer.from_defaults(
            token_limit=4096
        )

        # Create the system prompt
        self.system_prompt = "\n".join([role, goal, instructions, knowledge, llama_index_react_prompt])

        # Create the agent using the new Workflows API
        self.agent = ReActAgent(
            llm=self.llm,
            tools=self.tools,
            system_prompt=self.system_prompt,
            verbose=False
        )


    @staticmethod
    def date_tool():
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
            FunctionTool.from_defaults(
                fn=self.date_tool,
                name="date",
                description="Useful for getting the current date"
            ),
            FunctionTool.from_defaults(
                fn=self.web_search,
                name="web_search",
                description="Useful for searching the web for information"
            )
        ]

    async def _chat_async(self, message):
        """
        Async helper method for chat.

        Args:
            message (str): User's input message

        Returns:
            AgentOutput: The agent's response
        """
        handler = self.agent.run(user_msg=message, memory=self.memory)
        result = await handler
        return result

    def chat(self, message):
        """
        Send a message and get a response.

        Args:
            message (str): User's input message

        Returns:
            str: Assistant's response
        """
        try:
            # Try to use asyncio.run first (cleanest approach)
            try:
                result = asyncio.run(self._chat_async(message))
            except RuntimeError as e:
                # If there's already a running loop, use run_until_complete
                if "asyncio.run() cannot be called from a running event loop" in str(e):
                    loop = asyncio.get_event_loop()
                    result = loop.run_until_complete(self._chat_async(message))
                else:
                    raise

            # Extract the response text
            return str(result)

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
            # Reset the memory
            self.memory.reset()
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
