import os
import json
from datetime import datetime
import pandas as pd
import uuid
from openai import AzureOpenAI
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import Levenshtein

# FUNCTIONS

# Define the Open AI API response format
response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "uspto_classification_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "free_form_response": {
                    "type": "string",
                    "description": "The free-form text response providing detailed context or advice."
                },
                "classes": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "class_id": {
                                "type": "integer",
                                "description": "The ID of the USPTO class."
                            },
                            "class_name": {
                                "type": "string",
                                "description": "The name of the USPTO class."
                            },
                            "likelihood": {
                                "type": "string",
                                "enum": ["high", "medium", "low"],
                                "description": "Likelihood that this class is relevant."
                            }
                        },
                        "required": ["class_id", "class_name", "likelihood"],
                        "additionalProperties": False
                    }
                },
                "terms": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "term": {
                                "type": "string",
                                "description": "The term recommended for filing."
                            },
                            "class_id": {
                                "type": "integer",
                                "description": "The ID of the USPTO class this term belongs to."
                            },
                            "class_name": {
                                "type": "string",
                                "description": "The name of the USPTO class this term belongs to."
                            },
                            "likelihood": {
                                "type": "string",
                                "enum": ["high", "medium", "low"],
                                "description": "Likelihood that this term is relevant."
                            }
                        },
                        "required": ["term", "class_id", "class_name", "likelihood"],
                        "additionalProperties": False
                    }
                }
            },
            "required": ["free_form_response", "classes", "terms"],
            "additionalProperties": False
        }
    }
}



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


def load_id_manual(idmanual_path, classes_path):
    """
    Load the ID manual, clean up, and enrich it with class names from a lookup file.
    
    Parameters:
        idmanual_path (str): Path to the ID manual CSV file.
        classes_path (str): Path to the classes lookup CSV file.
    
    Returns:
        idm (pd.DataFrame): A cleaned and enriched DataFrame of the ID manual.
    """
    # Load ID manual and classes lookup
    idm_raw = pd.read_csv(idmanual_path, low_memory=False)
    classes_df = pd.read_csv(classes_path)
    classes_df['class_id'] = classes_df['class_id'].astype(str)
    
    # Merge class names into the ID manual
    idm = idm_raw.merge(classes_df, left_on="Class", right_on="class_id", how='left')
    idm.drop(columns=['Class'], inplace=True)

    # Clean up column names
    idm.columns = idm.columns.str.lower().str.replace(' ', '_').str.replace(r'[^\w\s]', '', regex=True)
    
    # Convert dates and clean specific columns
    idm['effective_date'] = pd.to_datetime(idm['effective_date'], errors='coerce')
    idm['ncl_version'] = idm['ncl_version'].str.replace('"', '', regex=False)
    
    # Reorder columns
    column_order = ['class_id', 'class_name', 'type', 'term_id', 'description', 
                    'ncl_version', 'status', 'effective_date', 'notes']
    idm = idm[column_order]
    
    # Remove logically deleted records
    idm = idm[idm['status'] != 'D']
    
    # Reset index
    idm.reset_index(drop=True, inplace=True)
    
    print(f"Loaded {len(idm)} records from ID manual.")

    return idm


def search_term(term, class_id, idm, sort_by="cosine_sim", max_nbr_terms_returned=10):
    """
    Search for a term in the IdManual and return similar terms based on cosine similarity and Levenshtein distance.

    Args:
    term (str): The USPTO term that will be searched.
    class_id (str): Restricts the search to terms in that class_id.
    idm (pd.DataFrame): Dataframe with USPTO id Manual, including terms ("description") and term embeddings.
    sort_by (str): "cosine_sim" or "levensthein". Indicate sort_order.
    max_nbr_terms_returned (int): Maximum number of terms to return.

    Returns:
    dict: JSON object with search results.
    """
    term = term.lower().strip()
    idm['description'] = idm['description'].str.lower().str.strip()

    if class_id:
        idm = idm[idm['class_id'] == class_id]

    try:
        term_emb = client.embeddings.create(input=[term], model="text-embedding-3-large").data[0].embedding
    except Exception as e:
        return {"error": f"Failed to create embedding for term: {str(e)}"}

    results = []
    for _, row in idm.iterrows():
        idm_term = row['description']
        idm_term_emb = row['description_emb']
        
        cosine_sim = cosine_similarity([term_emb], [idm_term_emb])[0][0]
        exact_match = term == idm_term
        leven_dist = Levenshtein.distance(term, idm_term)
        
        results.append({
            "term": idm_term,
            "cosine_sim": cosine_sim,
            "exact_match": exact_match,
            "levenshtein_distance": leven_dist
        })

    if sort_by == "cosine_sim":
        results = sorted(results, key=lambda x: x['cosine_sim'], reverse=True)
    elif sort_by == "levensthein":
        results = sorted(results, key=lambda x: x['levenshtein_distance'])

    search_results = {
        "original_term": term,
        "assumed_class": class_id,
        "exact_match_found": any(res['exact_match'] for res in results),
        "similar_terms": results[:max_nbr_terms_returned]
    }

    return {"search_results": [search_results]}


def create_search_results(gpt_draft_content, idm):
    """For each of the terms in the GPT initial response, search the ID Manual for similar terms and return the search results.

    Args:
    gpt_draft_content (dict): The initial (draft) response from GPT.
    idm (pd.DataFrame): The USPTO ID Manual DataFrame.

    Returns:
    search_results (list): A list of search results.
    """

    search_results = []
    for term_info in gpt_draft_content.get("terms", []):
        term = term_info["term"]
        class_id = str(term_info["class_id"])
        result = search_term(term, class_id, idm)
        search_results.append(result["search_results"][0])
    return search_results


# PROMPTS
def create_refinement_prompt(search_results):

    refinement_prompt = f'''   

    BACKGROUND:
    ----------
    The response you just wrote to the user will not be shown.
    Instead, to help you increase the quality of your response, we have compared your suggested terms (if any) with USPTO ID Manual by doing a similarity search  
    The search results are included below.

    YOUR TASK:
    ---------- 
    Use the search results to refine and finalize your previous answer, 
    ensuring that the goods/services align with official ID Manual terms whenever possible.
    Typically should should slightly revise the original draft. Use your judgement. 

    EXAMPLE OF TYPES OF CHANGES:
    ------------------------------------------------
    - **Modify terms** that do not exist in the ID Manual to an official ID Manual term or the closest match.
    - **Add terms** that might be more appropriate if the search results show similar or near-identical items.
    - **Use custom terms** (i.e., not in the ID Manual) only if none of the standard terms match the user's needs. This is less common. You need to motivate why standard terms may not be sufficient

    SEARCH RESULTS:
    ---------------
    The search results below holds similar terms for each term ("original_term") you initially mentioned.
    Search results are ranked by similarity to the original term.

    {search_results}'''

    return refinement_prompt


def analyze_initial_output(initial_output):
    """Hardcoded analysis logic for prototype."""
    return f"{initial_output} Additionally, consider Class 10 for medical devices."


# OPEN AI INITIALIZATION
#---------------------------------------------
deployment_name = "gpt-4o-mini"
api_version = "2024-08-01-preview"
azure_endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT_TESTAPP")
api_key = os.environ.get("AZURE_OPENAI_SECRET_KEY_TESTAPP")

client = AzureOpenAI(
    api_key=api_key,
    azure_endpoint=azure_endpoint,
    api_version=api_version
)

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

    # Fixed initial system prompt
    system_prompt = {
        "role": "system",
        "content": (
            "Provide expert advice on how to classify trademarks into USPTO classes. "
            "Advice the user which class (or classes), and which underlying terms fit his product (good or service)"
            "Make the initial output brief, not verbose."
            "Ask clarifying questions as needed to arrive at the exact classification. "
            " Typically ask one question at a time. Make it interactive."
            
        )
    }
    prompt_with_history = [system_prompt] + st.session_state["messages"]
    # First call to GPT. Generates a draft response that will not be visible to the user.
    try:
        gpt_draft_response = client.chat.completions.create(
            model=deployment_name,
            messages=initial_prompt
        )
        gpt_draft_content_response = gpt_draft_response.choices[0].message.content
    except Exception as e:
        st.error(f"Error in initial GPT call: {e}")
        gpt_draft_content_response = ""

    

    # OLD Analyze and modify output
    # analyzed_output = analyze_initial_output(initial_content)

    # Analyze the gpt draft response (For now: just Create search results)
    search_results = create_search_results(gpt_draft_content, idm)

    # Create refinement prompt. This includes the search results
    refinement_prompt = create_refinement_prompt(search_results)

    
    refinement_prompt_with_history = prompt_with_history + [
        {
            "role": "assistant",
            "content": gpt_draft_content_responsecontent
        },
        {
            "role": "system",
            "content": refinement_prompt
        }
    ]
 
    # Second GPT call
    try:
        gpt_final_response = client.chat.completions.create(
            model=deployment_name,
            messages=refinement_prompt
        )
        gpt_final_content_response = gpt_final_response.choices[0].message.content
    except Exception as e:
        st.error(f"Error in gpt_final_content_response call: {e}")
        gpt_final_content_response = "Error generating response."


    # Display the GPT final response to the user
    with st.chat_message("assistant"):
        st.markdown(gpt_final_content_response)

    # Append final GPT response to history. Note that the initial assistant message is NOT included.
    st.session_state["messages"].append(
        {"role": "assistant", "content": gpt_final_content_response}
    )

    # Save conversation to CSV
    save_messages_to_csv(
        messages=st.session_state["messages"],
        session_key=st.session_state["session_key"]
    )


