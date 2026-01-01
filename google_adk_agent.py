import os
import asyncio
import uuid
import json
import warnings
from datetime import date
from dotenv import load_dotenv
from tavily import TavilyClient

# --- Suppress Pydantic noise (ADK-specific) ---
warnings.filterwarnings("ignore", message=".*Pydantic serializer warnings.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

# Google ADK imports
from google.adk.agents import Agent as ADKAgent
from google.adk.sessions import InMemorySessionService
from google.adk.apps.app import App
from google.adk.runners import Runner
from google.genai import types

# Prompt components
from prompts import role, goal, instructions, knowledge

# Load environment variables
load_dotenv()

tavily_api_key = os.getenv("TAVILY_API_KEY")
tavily_client = TavilyClient(api_key=tavily_api_key)


class Agent:
    def __init__(self, model="gpt-4o-mini"):
        """
        Initialize the Google ADK agent.

        Args:
            model (str): The language model to use
        """
        self.name = "Google ADK Agent"

        self._instruction = "\n".join([role, goal, instructions, knowledge])

        self._adk_agent = ADKAgent(
            name="decision_support_agent",
            model=f"openai/{model}",
            instruction=self._instruction,
            tools=[self.date_tool, self.web_search]
        )

        self._app = App(
            name="AgentExamples",
            root_agent=self._adk_agent
        )

        self._session_service = InMemorySessionService()
        self._runner = Runner(
            app=self._app,
            session_service=self._session_service
        )

        self._user_id = "default_user"
        self._session_id = str(uuid.uuid4())

        # Create session eagerly (sync wrapper over async)
        asyncio.run(
            self._session_service.create_session(
                session_id=self._session_id,
                user_id=self._user_id,
                app_name=self._app.name
            )
        )

    # ------------------------------------------------------------------
    # Tools (EXACT identity + behavior)
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
        async def _run():
            response_text = ""

            content = types.Content(
                role="user",
                parts=[types.Part(text=message)]
            )

            async for event in self._runner.run_async(
                user_id=self._user_id,
                session_id=self._session_id,
                new_message=content
            ):
                if hasattr(event, "is_final_response") and event.is_final_response():
                    if hasattr(event, "text") and event.text:
                        response_text = event.text
                    elif hasattr(event, "content") and hasattr(event.content, "parts"):
                        response_text = "".join(
                            p.text for p in event.content.parts
                            if hasattr(p, "text") and p.text
                        )

            return response_text

        try:
            result = asyncio.run(_run())
            return result if result else "Agent generated an empty response."
        except Exception as e:
            return f"System Error: {str(e)}"

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
            self._session_id = str(uuid.uuid4())
            asyncio.run(
                self._session_service.create_session(
                    session_id=self._session_id,
                    user_id=self._user_id,
                    app_name=self._app.name
                )
            )
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

