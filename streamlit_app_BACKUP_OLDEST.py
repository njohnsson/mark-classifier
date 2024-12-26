import os
from datetime import datetime
import pandas as pd
import uuid
from openai import AzureOpenAI
import streamlit as st

# FUNCTIONS
# -----------------------------------------------------------------------------
def save_messages_to_csv(messages, session_key, filename="chat_history.csv"):
    # Add the session key and timestamp to each message
    messages_with_session_and_time = [
        {
            "session_key": session_key,
            "timestamp": msg.get("timestamp", datetime.now().isoformat()),
            "role": msg["role"],
            "content": msg["content"]
        }
        for msg in messages
    ]
    # Convert messages to a pandas DataFrame
    df = pd.DataFrame(messages_with_session_and_time)
    # Write DataFrame to CSV file. Append (a) if file exists, else write header.
    df.to_csv(filename, mode='a', header=not os.path.isfile(filename), index=False, encoding='utf-8')

# INITIALIZATIONS
# -----------------------------------------------------------------------------

# Get Model and Open AI client info
deployment_name = "gpt-4o-mini"
api_version = "2024-08-01-preview"
azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT_TESTAPP")
api_key = os.environ.get("AZURE_OPENAI_SECRET_KEY_TESTAPP")


client = AzureOpenAI(
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version
)


st.title("ChatGPT-Niklas")

# Generate a unique session key. Used to identify session in chat history csv
if "session_key" not in st.session_state:
    st.session_state["session_key"] = str(uuid.uuid4())



if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = deployment_name

# For QA
print("azure_endpoint:", api_key)
print("MODEL:", deployment_name)

# Define system prompt content inline (instead of using prompts library)
system_prompt = {
    "role": "system",
    "content": "You are ChatGPT, a helpful assistant that follows the user's instructions carefully."
}

# Define an assistant intro message (instead of using prompts library)
assistant_intro_message = "Hello! I'm your assistant. How can I help you today?"

# If this is the start of the session, initialize messages
# Exclude the system prompt from the displayed messages
if "messages" not in st.session_state:
    st.session_state["messages"] = [
        {"role": "user", "content": assistant_intro_message}
    ]


# CONDUCT ONE CHAT INTERACTION (1. User input, 2. Model response)
# -----------------------------------------------------------------------------
# Display message history
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# For QA
print(st.session_state["messages"])

# Display text box for user input
prompt = st.chat_input("What is up?")

# Once user has submitted a prompt
if prompt:
    # Add user message to the session state messages
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display the user's message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build full message history to send to the model (include system prompt)
    message_history_incl_system_prompt = [system_prompt] + st.session_state.messages

    with st.chat_message("assistant"):
        responses = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": m["role"], "content": m["content"]}
                for m in message_history_incl_system_prompt
            ],
            stream=True,
        )

        full_response_text = ""
        try:
            for response_chunk in responses:
                if response_chunk.choices and response_chunk.choices[0].delta.content is not None:
                    full_response_text += response_chunk.choices[0].delta.content

            st.markdown(full_response_text)
            st.session_state.messages.append({"role": "assistant", "content": full_response_text})
        except Exception as e:
            st.error(f"Failed to process response: {e}")

        # Save the updated conversation to CSV
        save_messages_to_csv(
            messages=st.session_state.messages,
            session_key=st.session_state["session_key"]
        )
