import os
import openai
import json
from tavily import TavilyClient
from dotenv import load_dotenv
from datetime import date
from prompts import role, goal, instructions, knowledge

# Load environment variables
load_dotenv()

tavily_api_key = os.getenv("TAVILY_API_KEY")
openai_api_key = os.getenv("OPENAI_API_KEY")
tavily_client = TavilyClient(api_key=tavily_api_key)


class Agent:
    def __init__(self, model="gpt-4o-mini"):
        """
        Initialize the OpenAI agent using the Responses API.

        Args:
            model (str): The language model to use
        """
        self.name = "OpenAI Responses API Agent"
        self.model = model
        self.client = openai.OpenAI(api_key=openai_api_key)
        self.previous_response_id = None

        # System instructions
        self.system_instructions = "\n".join([role, goal, instructions, knowledge])

    ### Tools ###
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

    def _prepare_tools(self):
        """
        Prepare tool definitions for the Responses API.
        """
        return [
            {
                "type": "function",
                "name": "date",
                "description": "Get the current date",
                "parameters": {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            },
            {
                "type": "function",
                "name": "web_search",
                "description": "Search the web for information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        }
                    },
                    "required": ["query"]
                }
            }
        ]

    def _call_tool(self, tool_name, tool_input):
        """
        Call the appropriate tool based on the tool name.

        Args:
            tool_name (str): Name of the tool to call
            tool_input (dict): Input parameters for the tool

        Returns:
            str: Tool output
        """
        if tool_name == "date":
            return self.date_tool()
        elif tool_name == "web_search":
            return self.web_search(tool_input.get("query", ""))
        else:
            return "Unsupported tool."

    def _process_response(self, response, input_items):
        """
        Process a response and handle any tool calls.

        Args:
            response: The response object from the API
            input_items: Current conversation input items

        Returns:
            tuple: (final_response_text, new_previous_response_id, updated_input_items)
        """
        # Check if there are tool calls to handle
        tool_calls_found = False

        # Look through output items for tool calls
        for output_item in response.output:
            if hasattr(output_item, 'type') and output_item.type == 'function_call':
                tool_calls_found = True
                break

        if not tool_calls_found:
            # Extract text response
            response_text = getattr(response, 'output_text', '')
            return response_text, response.id, input_items

        # Handle tool calls
        tool_results = []
        for output_item in response.output:
            if hasattr(output_item, 'type') and output_item.type == 'function_call':
                function_name = output_item.name
                try:
                    function_args = json.loads(output_item.arguments) if output_item.arguments else {}
                except json.JSONDecodeError:
                    function_args = {}

                # Execute the tool
                result = self._call_tool(function_name, function_args)

                # Add function call result to items
                tool_results.append({
                    "type": "function_call_output",
                    "call_id": output_item.call_id,
                    "output": result
                })

        # Make another call with the tool results
        follow_up_response = self.client.responses.create(
            model=self.model,
            instructions=self.system_instructions,
            input=tool_results,
            previous_response_id=response.id,
            tools=self._prepare_tools()
        )

        # Extract final response text
        final_text = getattr(follow_up_response, 'output_text', '')

        # Update input items to include tool results for conversation continuity
        updated_items = input_items + tool_results

        return final_text, follow_up_response.id, updated_items

    def chat(self, message):
        """
        Send a message and get a response.

        Args:
            message (str): User's input message

        Returns:
            str: Assistant's response
        """
        try:
            # Prepare input - use simple message format
            input_item = {
                "role": "user",
                "content": message
            }

            # Create response
            if self.previous_response_id:
                # Continue previous conversation
                response = self.client.responses.create(
                    model=self.model,
                    instructions=self.system_instructions,
                    input=[input_item],
                    previous_response_id=self.previous_response_id,
                    tools=self._prepare_tools()
                )
            else:
                # Start new conversation
                response = self.client.responses.create(
                    model=self.model,
                    instructions=self.system_instructions,
                    input=[input_item],
                    tools=self._prepare_tools()
                )

            # Process response and handle tool calls
            response_text, new_response_id, _ = self._process_response(response, [input_item])

            # Update previous_response_id for conversation continuity
            self.previous_response_id = new_response_id

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
            self.previous_response_id = None
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
