import os
import json
import warnings
from datetime import date

from dotenv import load_dotenv
from tavily import TavilyClient

# Suppress Pydantic serialization warnings
warnings.filterwarnings("ignore", message=".*Pydantic serializer warnings.*")
warnings.filterwarnings("ignore", message=".*Valid config keys have changed.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# SmolAgents imports
from smolagents import ToolCallingAgent, tool, LiteLLMModel, LogLevel

from prompts import role, goal, instructions, knowledge

# Load environment variables
load_dotenv()

tavily_api_key = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=tavily_api_key)


class Agent:
    def __init__(self, model="gpt-4o-mini"):
        """
        Initialize the SmolAgents agent.

        Args:
            model (str): The language model to use
        """
        self.name = "SmolAgents Agent"

        # System instructions
        agent_instructions = "\n".join([role, goal, instructions, knowledge])

        # Create model instance
        llm_model = LiteLLMModel(model_id=model)

        # Create agent with tools
        self._agent = ToolCallingAgent(
            tools=[self._create_date_tool(), self._create_web_search_tool()],
            model=llm_model,
            instructions=agent_instructions,
            verbosity_level=LogLevel.OFF
        )

    # ------------------------------------------------------------------
    # Required static tools
    # ------------------------------------------------------------------

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
        search_response = tavily_client.search(query)
        results = json.dumps(search_response.get("results", []))
        print(f"Web Search Results for '{query}':")
        print(results)
        return results

    # ------------------------------------------------------------------
    # SmolAgents tool wrappers
    # ------------------------------------------------------------------

    def _create_date_tool(self):
        """Create date tool using @tool decorator"""
        @tool
        def date_tool() -> str:
            """Get the current date."""
            return Agent.date_tool()
        return date_tool

    def _create_web_search_tool(self):
        """Create web search tool using @tool decorator"""
        @tool
        def web_search(query: str) -> str:
            """Search the web for information.

            Args:
                query: The search query string
            """
            return Agent.web_search(query)
        return web_search

    # ------------------------------------------------------------------
    # Chat
    # ------------------------------------------------------------------

    def chat(self, message):
        """
        Send a message and get a response.

        Args:
            message (str): User's input message

        Returns:
            str: Assistant's response
        """
        try:
            # Run agent with task, don't reset to maintain conversation
            response = self._agent.run(task=message, reset=False)

            return str(response)
        except Exception as e:
            return f"System Error: {str(e)}"

    # ------------------------------------------------------------------
    # Clear Chat
    # ------------------------------------------------------------------

    def clear_chat(self):
        """
        Reset the conversation context.

        Returns:
            bool: True if reset was successful
        """
        try:
            self._agent.memory.reset()
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
        if query.lower() in ["exit", "quit"]:
            break

        response = agent.chat(query)
        print(f"Assistant: {response}")


if __name__ == "__main__":
    main()

