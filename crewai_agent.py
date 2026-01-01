import os
from dotenv import load_dotenv
from datetime import date
from tavily import TavilyClient
import json

# CrewAI imports (1.0+)
from crewai import Agent as CrewAIAgent
from crewai import Task, Crew
from crewai.tools import BaseTool

# Prompt components
from prompts import role, goal, instructions, knowledge

# Load environment variables
load_dotenv()

# Initialize Tavily client
tavily_api_key = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=tavily_api_key)

class _DateTool(BaseTool):
    name: str = "Get Current Date"
    description: str = "Function to get the current date."

    def _run(self):
        return Agent.date_tool()


class _WebSearchTool(BaseTool):
    name: str = "Web Search"
    description: str = "Search the web for a query and return results."

    def _run(self, query: str):
        return Agent.web_search(query)


class Agent:
    def __init__(self, model="gpt-4o-mini"):
        """
        Initialize the CrewAI agent.

        Args:
            model (str): The language model to use
        """
        self.name = "CrewAI Agent"

        self._tools = self._create_tools()

        self._agent = CrewAIAgent(
            role=role,
            goal="\n".join([goal, instructions]),
            backstory=knowledge,
            tools=self._tools,
            llm=model,
            verbose=False
        )

        # Explicit conversation history (framework-agnostic)
        self._messages = []

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

    def _create_tools(self):
        return [_DateTool(), _WebSearchTool()]



    def chat(self, message):
        """
        Send a message and get a response.

        Args:
            message (str): User's input message

        Returns:
            str: Assistant's response
        """
        try:
            task = Task(
                description=(
                    "Answer the user's query using the provided tools when helpful.\n\n"
                    "Conversation history:\n{history}\n\n"
                    "User query:\n{query}"
                ),
                expected_output=(
                    "A clear, well-structured response that directly answers the user's query."
                ),
                agent=self._agent
            )

            crew = Crew(
                agents=[self._agent],
                tasks=[task],
                verbose=False
            )

            response = crew.kickoff(
                inputs={
                    "query": message,
                    "history": self._messages
                }
            )

            # Update history
            self._messages.append({"role": "user", "content": message})
            self._messages.append({"role": "assistant", "content": str(response)})

            return str(response)

        except Exception as e:
            print(f"Error in chat: {e}")
            return "System Error: Unable to process request."

    # ------------------------------------------------------------------
    # Clear chat
    # ------------------------------------------------------------------

    def clear_chat(self):
        """
        Reset the conversation context.

        Returns:
            bool: True if reset was successful
        """
        try:
            self._messages = []
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
        if query.lower() == 'exit':
            break

        response = agent.chat(query)
        print(f"Assistant: {response}")


if __name__ == "__main__":
    main()
