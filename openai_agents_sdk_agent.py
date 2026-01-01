import os
from dotenv import load_dotenv
from datetime import date
from tavily import TavilyClient
import json
from agents import Agent as SDKAgent, function_tool, Runner
from prompts import role, goal, instructions, knowledge

# Load environment variables
load_dotenv()

# Initialize Tavily client
tavily_api_key = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=tavily_api_key)


class Agent:
    def __init__(self, model="gpt-4o-mini"):
        """
        Initialize the OpenAI Agents SDK agent.

        Args:
            model (str): The language model to use
        """
        self.name = "OpenAI Agents SDK Agent"
        self.model = model
        self.runner = Runner()
        self.conversation_history = []

        # Combine system instructions
        self.system_instructions = "\n".join([role, goal, instructions, knowledge])

        # Create the SDK agent with tools
        self.sdk_agent = SDKAgent(
            name="Decision Support Agent",
            instructions=self.system_instructions,
            model=self.model,
            tools=[self._create_date_tool(), self._create_web_search_tool()]
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
        search_response = tavily_client.search(query)
        results = json.dumps(search_response.get('results', []))
        print(f"Web Search Results for '{query}':")
        print(results)
        return results

    def _create_date_tool(self):
        """Create the date tool for the SDK agent."""
        @function_tool
        def date():
            """Get the current date."""
            return self.date_tool()
        return date

    def _create_web_search_tool(self):
        """Create the web search tool for the SDK agent."""
        @function_tool
        def web_search(query: str):
            """
            Search the web for information.

            Args:
                query: The search query string
            """
            return Agent.web_search(query)
        return web_search

    def chat(self, message):
        """
        Send a message and get a response.

        Args:
            message (str): User's input message

        Returns:
            str: Assistant's response
        """
        try:
            # Add user message to conversation history
            self.conversation_history.append({"role": "user", "content": message})

            # Run the agent with conversation history
            result = self.runner.run_sync(
                starting_agent=self.sdk_agent,
                input=self.conversation_history
            )

            # Extract the final output
            response_text = str(result.final_output)

            # Add assistant response to conversation history
            self.conversation_history.append({"role": "assistant", "content": response_text})

            return response_text

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
            self.conversation_history = []
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
