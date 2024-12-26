import os
from datetime import datetime
import pandas as pd
import uuid
from openai import AzureOpenAI
import streamlit as st

# FUNCTIONS
def save_messages_to_csv(messages, session_key, filename="chat_history.csv"):
    messages_with_session_and_time = [
        {
            "session_key": session_key,
            "timestamp": msg.get("timestamp", datetime.now().isoformat()),
            "role": msg["role"],
            "content": msg["content"]
        }
        for msg in messages
    ]
    df = pd.DataFrame(messages_with_session_and_time)
    df.to_csv(filename, mode='a', header=not os.path.isfile(filename), index=False, encoding='utf-8')

def analyze_initial_output(initial_output):
    return f"{initial_output}  At the beginning of your response, mention that this service is free of charge."

# INITIALIZATIONS
deployment_name = "gpt-4o-mini"
api_version = "2024-08-01-preview"
azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT_TESTAPP")
api_key = os.environ.get("AZURE_OPENAI_SECRET_KEY_TESTAPP")

client = AzureOpenAI(
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version
)

st.title("AI Mark Classifier - Prototype")

if "session_key" not in st.session_state:
    st.session_state["session_key"] = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state["messages"] = []

system_prompt = {
    "role": "system",
    "content": (
        "Provide advice on how to classify trademarks into USPTO classes. "
        "Tell the user which class (or classes) and underlying terms fit the good or service that the user wants to register. "
        "Make it interactive and ask clarifying questions as needed."
    )
}

for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input("Enter your trademark-related query++...")

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # First GPT call
    initial_prompt = [system_prompt] + st.session_state["messages"]
    try:
        initial_response = client.chat.completions.create(
            model=deployment_name,
            messages=initial_prompt
        )
        initial_content = initial_response.choices[0].message.content
    except Exception as e:
        st.error(f"Error in initial GPT call: {e}")
        initial_content = ""

    # Analyze and modify output
    analyzed_output = analyze_initial_output(initial_content)

    # Second GPT call
    refinement_prompt = initial_prompt + [
        {"role": "assistant", "content": initial_content},
        {"role": "system", "content": f"Revise based on this analysis: {analyzed_output}"}
    ]
    try:
        final_response = client.chat.completions.create(
            model=deployment_name,
            messages=refinement_prompt
        )
        final_content = final_response.choices[0].message.content
    except Exception as e:
        st.error(f"Error in refinement GPT call: {e}")
        final_content = "Error generating response."

    with st.chat_message("assistant"):
        st.markdown(final_content)

    st.session_state.messages.append({"role": "assistant", "content": final_content})

    save_messages_to_csv(
        messages=st.session_state["messages"],
        session_key=st.session_state["session_key"]
    )
