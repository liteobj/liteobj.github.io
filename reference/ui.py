import streamlit as st
import requests
import json
import urllib3
import threading
import time
from sse_starlette.sse import EventSourceResponse

# Disable SSL warnings
urllib3.disable_warnings()

# Configure the page
st.set_page_config(page_title="MCP Chat Interface", layout="wide")

# Initialize session state for chat history and notifications
if "messages" not in st.session_state:
    st.session_state.messages = []
if "notifications" not in st.session_state:
    st.session_state.notifications = []

# Title
st.title("MCP Chat Interface")

# Sidebar for API configuration and notifications
with st.sidebar:
    st.header("Configuration")
    api_url = st.text_input("API URL", "http://localhost:8000")
    
    st.header("Notifications")
    notification_placeholder = st.empty()
    
    # Function to update notifications
    def update_notifications():
        try:
            response = requests.get(
                f"{api_url}/notifications",
                stream=True,
                headers={"Accept": "text/event-stream"},
                verify=False
            )
            
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        message = data.get("message", "")
                        if message:
                            st.session_state.notifications.append(message)
                            if len(st.session_state.notifications) > 10:  # Keep last 10 notifications
                                st.session_state.notifications.pop(0)
                            st.rerun()
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"Error in notification thread: {e}")

    # Start notification thread
    if "notification_thread" not in st.session_state:
        st.session_state.notification_thread = threading.Thread(
            target=update_notifications,
            daemon=True
        )
        st.session_state.notification_thread.start()

# Display notifications
with notification_placeholder.container():
    st.subheader("Recent Notifications")
    for notification in reversed(st.session_state.notifications):
        st.info(notification)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to know?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Create a placeholder for the assistant's response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        try:
            # Make request to API with streaming
            response = requests.post(
                f"{api_url}/query",
                json={"query": prompt},
                stream=True,
                headers={"Accept": "text/event-stream"},
                verify=False
            )
            
            # Process streaming response
            for line in response.iter_lines():
                if line:
                    try:
                        data = json.loads(line.decode('utf-8'))
                        if data.get("event") == "message":
                            chunk = data.get("data", {}).get("chunk", "")
                            full_response += chunk + "\n"
                            message_placeholder.markdown(full_response + "▌")
                        elif data.get("event") == "error":
                            error = data.get("data", {}).get("error", "Unknown error")
                            st.error(f"Error: {error}")
                            break
                    except json.JSONDecodeError:
                        continue
            
            # Update the placeholder with the final response
            message_placeholder.markdown(full_response)
            
            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"Error connecting to API: {str(e)}")

# Add a clear chat button
if st.button("Clear Chat"):
    st.session_state.messages = []
    st.rerun() 