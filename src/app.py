import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import uuid
import streamlit as st


# from src.services.id_manual import load_id_manual
from src.services.id_manual import load_idm, create_search_results
from src.services.model_api import initialize_openai_client, get_gpt_response, get_response_format
from src.services.utils import save_conversation
from src.prompts.prompts import create_system_prompt, create_refinement_prompt

# MISC. INITIALIZATION
#---------------------------------------------
openai_client = initialize_openai_client()
idm = load_idm() # Load ID Manual dataframe

# STREAMLIT INITIALIZATION
#---------------------------------------------
# Create unique session key
if "session_key" not in st.session_state:
    st.session_state["session_key"] = str(uuid.uuid4())

# Keep track of user-assistant conversation
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Layout: single column for chat (or you can do st.columns if you like)
st.title("AI Mark Classifier")

# Display chat messages
for message in st.session_state["messages"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input box for user queries
prompt = st.chat_input("Describe the product that you want to trademark...")

# STREAMLIT INTERACTION
#---------------------------------------------

# Whenever user has entered a prompt
if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

   # Draft call to GPT. The response will NOT be visible to the user.
    prompt_with_history = [create_system_prompt()] + st.session_state["messages"] 
    gpt_draft_content_response, _ = get_gpt_response(openai_client, prompt_with_history, get_response_format(require_term_id=False))
 
    print("GPT DRAFT RESPONSE")
    print(type(gpt_draft_content_response))
    print(gpt_draft_content_response)

    # TEMP: Ensure term_id is present in the response
    # if 'term_id' not in gpt_draft_content_response:
    #    gpt_draft_content_response['term_id'] = None
    
    # Analyze the gpt draft response (For now: just Create search results)
    search_results = create_search_results(gpt_draft_content_response, idm, openai_client)

    # Create refinement prompt.
    # TO-DO!!!!!: remove the assistant message. Put the content in the system message instead. 
    refinement_prompt_content = create_refinement_prompt(search_results)
    refinement_prompt_with_history = prompt_with_history + [
        {"role": "assistant",
         "content": gpt_draft_content_response["free_form_response"]
        },
        {
            "role": "system",
            "content": refinement_prompt_content
        }
    ]
 
    # Second GPT call
    gpt_final_content_response, _ = get_gpt_response(openai_client, refinement_prompt_with_history, get_response_format(require_term_id=True))


    # Display the free-form part of GPT final response to the user
    with st.chat_message("assistant"):
        st.markdown(gpt_final_content_response["free_form_response"])

    # Append final GPT response to history. Note that the initial assistant message is NOT included.
    st.session_state["messages"].append(
        {"role": "assistant", "content": gpt_final_content_response["free_form_response"]}
    )

    # To Add: Grid with search results

    # Save conversation to CSV
    save_conversation(
        messages=st.session_state["messages"],
        session_key=st.session_state["session_key"]
    )


#QA individual functions here.
