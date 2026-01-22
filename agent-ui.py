import streamlit as st
import importlib
import os
from types import ModuleType

# Fix annoying UI issues
st.markdown(
    """
    <style>
    .stAppDeployButton {
        visibility: hidden;
    }
    .stSidebar {
            min-width: 200px;
            max-width: 200px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

def _is_valid_agent_instance(obj) -> bool:
    if obj is None:
        return False
    if not hasattr(obj, "name"):
        return False
    if not hasattr(obj, "chat") or not callable(getattr(obj, "chat", None)):
        return False
    return True

def _safe_get_agent_instance(module: ModuleType):
    AgentClass = getattr(module, "Agent", None)
    if AgentClass is None or not callable(AgentClass):
        return None
    try:
        inst = AgentClass()
    except Exception:
        return None
    return inst if _is_valid_agent_instance(inst) else None

# Function to get available agent modules and their names
def get_available_agents():
    agents = {}
    for file in os.listdir('.'):
        if file.endswith('_agent.py'):
            module_name = file[:-3]  # Remove .py
            try:
                module = importlib.import_module(module_name)
                temp_agent = _safe_get_agent_instance(module)
                if temp_agent is None:
                    raise AttributeError("Invalid Agent implementation")
                agents[module_name] = temp_agent.name
            except Exception as e:
                print(f"Error loading {module_name}: {str(e)}")
    return agents

# Add agent selector to sidebar
available_agents = get_available_agents()

if not available_agents:
    st.sidebar.warning("No agents found. Please add an *_agent.py module.")
    st.stop()

selected_agent = st.sidebar.selectbox(
    "Select Agent Type",
    options=list(available_agents.keys()),
    format_func=lambda x: available_agents[x],
    key="agent_selector"
)

# Dynamic import of selected agent
if "current_agent_type" not in st.session_state:
    st.session_state.current_agent_type = selected_agent

# If agent type changed, reset the session
if st.session_state.current_agent_type != selected_agent:
    st.session_state.current_agent_type = selected_agent
    if "agent" in st.session_state:
        del st.session_state.agent
    if "messages" in st.session_state:
        st.session_state.messages = []

# Initialize agent
if "agent" not in st.session_state:
    try:
        module = importlib.import_module(selected_agent)
        agent_instance = _safe_get_agent_instance(module)
        if agent_instance is None:
            raise AttributeError("Invalid Agent implementation")
        st.session_state.agent = agent_instance
    except Exception as e:
        # 🔒 VOTAL.AI Security Fix: Verbose exception disclosure to end users reveals internal details [CWE-209] - LOW
        st.error("Error loading agent.")  # Do not disclose internal details

if "agent" not in st.session_state:
    st.stop()

# Store chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display title with agent name
st.title(f"Chat with {st.session_state.agent.name}")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
if user_input := st.chat_input("Type your message..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get response from agent
    with st.chat_message("assistant"):
        response_container = st.empty()
        response_text = ""

        try:
            response = st.session_state.agent.chat(user_input)
            response_text = str(response)
        except Exception as e:
            response_text = "Error: Something went wrong."  # Do not disclose internal details

        response_container.markdown(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})

# Clear chat button
if st.sidebar.button("Clear Chat"):
    if hasattr(st.session_state.agent, "clear_chat") and callable(getattr(st.session_state.agent, "clear_chat", None)):
        try:
            st.session_state.agent.clear_chat()
        except Exception:
            pass
    st.session_state.messages = []
    st.rerun()