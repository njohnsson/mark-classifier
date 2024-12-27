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
def get_response_format(require_term_id=False):
    schema = {
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
                        "term_id": {
                            "type": "string",
                            "description": "If available. The ID of the term, e.g. '035-2814'"
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
                            "description": "Likelihood that the user should include this term."
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

    if require_term_id:
        required_fields = schema["properties"]["terms"]["items"]["required"]
        if "term_id" not in required_fields:
            required_fields.append("term_id")
    else:
        del schema["properties"]["terms"]["items"]["properties"]["term_id"]

    return {
        "type": "json_schema",
        "json_schema": {
            "name": "uspto_classification_response",
            "strict": True,
            "schema": schema
        }
    }

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


def search_one_term(search_term, class_id, idm, sort_by="cosine_sim", max_nbr_terms_returned=10):
    """
    Search for a term in the IdManual and return similar terms based on cosine similarity and Levenshtein distance.

    Args:
    term (str): The USPTO term that will be searched.
    class_id (str): Restricts the search to terms in that class_id.
    idm (pd.DataFrame): Dataframe with USPTO id Manual
    sort_by (str): "cosine_sim" or "levensthein". Indicate sort_order.
    max_nbr_terms_returned (int): Maximum number of terms to return.

    Returns:
    search_result (dict): dict with search results.
    """

    # Prepare the term for search, and get its embedding
    search_term = search_term.lower().strip()
    try:
        search_term_emb = client.embeddings.create(input=[search_term], model="text-embedding-3-large").data[0].embedding
    except Exception as e:
        return {"error": f"Failed to create embedding for search term: {str(e)}"}


    idm_filtered = idm.loc[idm.class_id==class_id,:].copy()
    if idm_filtered.empty:
        return {"error": f"No terms found for class_id: {class_id}"}

    #Merge the embeddings to the filtered idm
    class_id_str = str(class_id).zfill(2)
    idm_term_embeddings = pd.read_pickle(f'./data/idm_embeddings/idm_embeddings_class_{class_id_str}.pkl')

    # Merge the embeddings to the filtered idm
    # Verify that the idm term embeddings match the filtered idm dataframe indices
    assert len(idm_filtered) == len(idm_term_embeddings)
    assert idm_filtered.index.equals(idm_term_embeddings.index)
    idm_term_embeddings.name = 'idm_term_embedding' # Will name the df column
    idm_filtered_emb = idm_filtered.merge(idm_term_embeddings, left_index=True, right_index=True)
    assert len(idm_filtered) == len(idm_filtered_emb)


    # Compare similarity of the search term to each term in the filtered ID Manual
    results = []
    for _, row in idm_filtered_emb.iterrows():
        
        # Unpack
        idm_term = row['description']
        idm_term_lower = idm_term.lower().strip()
        idm_term_embedding = row['idm_term_embedding']
        idm_term_id = row['term_id']
        
        # Calculate similarity metrics
        exact_match = (search_term == idm_term_lower)
        leven_dist = Levenshtein.distance(search_term, idm_term_lower)
        cosine_sim = cosine_similarity([search_term_emb], [idm_term_embedding])[0][0]
        
        # Append results
        results.append({
            "term": idm_term, # Note: Output original case although comparison is in lowercase
            "term_id": idm_term_id,
            "exact_match": exact_match,
            "levenshtein_distance": leven_dist,
            "cosine_sim": cosine_sim
        })

    if sort_by == "cosine_sim":
        results = sorted(results, key=lambda x: x['cosine_sim'], reverse=True)
    elif sort_by == "levensthein":
        results = sorted(results, key=lambda x: x['levenshtein_distance'])

    search_results = {
        "original_term": search_term,
        "assumed_class": class_id,
        "exact_match_found": any(res['exact_match'] for res in results),
        "similar_terms": results[:max_nbr_terms_returned]
    }

    return {"search_results": [search_results]}


def create_search_results(gpt_draft_content, idm, max_nbr_terms_returned=10):
    """For each of the terms in the GPT initial response, search the ID Manual for similar terms and return the search results.

    Args:
    gpt_draft_content (dict): The initial (draft) response from GPT.
    idm (pd.DataFrame): The USPTO ID Manual DataFrame.

    Returns:
    search_results (list): A list of search results.
    """

    assert type(gpt_draft_content) == dict

    search_results = []
    for term_info in gpt_draft_content.get("terms", []):
        term = term_info["term"]
        class_id = str(term_info["class_id"])
        result = search_one_term(term, class_id, idm, max_nbr_terms_returned=max_nbr_terms_returned)
        search_results.append(result["search_results"][0])
    return search_results


# PROMPTS
def create_refinement_prompt(search_results):

    refinement_prompt = f'''   

    BACKGROUND:
    ----------
    - The response you just wrote to the user will not be shown. Consider it a draft.
    - To help you increase the response quality, 
      your suggested terms (if any) were compared with the pre-approved terms in the USPTO ID Manual to 1) see if they exist, and 
      2) to find other similar terms that you can suggest to the user.  


    YOUR TASK:
    ---------- 
    - Use the search results to modify and refine your previous answer.
    - Ensure that each term you suggest matches pre-approved ID Manual terms.
    - For each term, also provide the ID Manual term ID (e.g. "ID: 009-0481"), as provided by the search results. If you suggest a custom term say "(custom)".
    - Use your judgement when deciding how to incorporate the search results into your response.
      Often, it is enough to slightly revise the original message.


    EXAMPLE OF TYPES OF CHANGES:
    ----------------------------
    - **Modify term**: Modify the initial term to exactly match a (highly similar) term in the language ID Manual.
    - **Add terms**: if search results reveals that there are many highly similar terms, you may need to output them and then ask the user clarifying questions to understand which one(s) to use. 
    - **Use custom terms**: In some (unusual) cases none of the Id Manual terms match the user's needs, and you may recommend that the user applies with a custom term. If so, explain why.

    SEARCH RESULTS:
    ---------------
    The search results below holds similar terms for each term ("original_term") you initially mentioned.
    Search results are ranked by similarity to the original term.

    {search_results}

    OUTPUT FORMAT:
    --------------
    - In general, use a free-form text response, just like in your draft response.
    - If there are more than one term for a class, typically use a bullet list to list terms.
    - Class Id and names should be bold.
    - Terms should be in italics.
    - After each term name, add the Manual ID within parenthesis such as "(ID: 009-0481)". If you suggest a custom term say "(custom)".

    EXAMPLE OF CONVERSATION:
    -------------------------
    USER: I have developed a harnesss for bears.
    ASSISTANT: This could fall into **Class 18 - Leather goods**, especially the term _harnesses for animals_.
    It may also be relevant to **Class 28 - Toys and sporting goods**.
    Please provide some more details. What is the material? In what situations would the harness would be used?
    User: The harness is made of leather and nylon. It will be used by researchers study bear behavior.
    The harness lets researchers mount a camera on the bear as well as GPS tracking devices.
    ASSISTANT: Given that, these classes and pre-approved terms are relevant:
    - **Class 9 - Scientific and electronic equipment**:
        - _GPS tracking devices (ID: 009-0475)_
        - _Scientific research equipment (ID: 009-0481)_
    - **Class 18 - Leather goods**:
        - _harnesses for animals (ID: 018-0321)_
        - harness for bears (ID: 018-0407)_
    Besides researchers, could the harness also be used by other groups of people owning bears? If so class 28 may also be relevant.
    
    <Conversation continues>
    


    '''

    return refinement_prompt


# def analyze_initial_output(initial_output):
#    """Hardcoded analysis logic for prototype."""
#    return f"{initial_output} Additionally, consider Class 10 for medical devices."



def save_messages_to_csv(messages, session_key, filename="chat_history.csv"):
    messages_with_session_and_time = [
        {
            "session_key": session_key,
            "timestamp": msg.get("timestamp", datetime.now().isoformat()),
            "role": msg["role"],
            "content": msg["content"]
        }
        for msg in messages if msg["role"] in ["user", "assistant", "system"]
    ]
    df = pd.DataFrame(messages_with_session_and_time)
    df.to_csv(filename, mode='a', header=not os.path.isfile(filename), index=False, encoding='utf-8')


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
            "You are a trademark expert providing advice on how to classify trademarks for a USPTO application. "
            "Advise the user which class(es), and which pre-approved underlying terms fit their product. "
            "Make the initial output brief, not verbose. "
            "Ask clarifying questions as needed to arrive at the exact classification. "
            "Typically ask one question at a time. Make it interactive. "
            "The output response format is JSON where the 'free_form_response' field will be displayed to the user. "
            "In addition, in the JSON output you explicitly specify classes, terms, and the likelihood that a term is applicable. "
            "\n\n"
            "EXAMPLE OF CONVERSATION, SHOWING ONLY THE FREE-FORM RESPONSE PORTION:\n"
            "---------------------------------------------------------------------\n"
            "USER: I have developed a harness for bears.\n"
            "ASSISTANT: This could fall into **Class 18 - Leather goods**, especially the term _harnesses for animals_.\n"
            "It may also be relevant to **Class 28 - Toys and sporting goods**.\n"
            "Please provide some more details. What is the material? In what situations would the harness be used?\n"
            "USER: The harness is made of leather and nylon. It will be used by researchers studying bear behavior.\n"
            "The harness lets researchers mount a camera on the bear as well as GPS tracking devices.\n"
            "ASSISTANT: Given that, these classes and pre-approved terms are relevant:\n"
            "- **Class 9 - Scientific and electronic equipment**:\n"
            "    - _GPS tracking devices_\n"
            "    - _Scientific research equipment_\n"
            "- **Class 18 - Leather goods**:\n"
            "    - _harnesses for animals_\n"
            "    - _harness for bears_\n"
            "Besides researchers, could the harness also be used by other groups of people owning bears? If so, class 28 may also be relevant.\n"
            "\n"
            "<Conversation continues>\n"
        )
    }
 
    prompt_with_history = [system_prompt] + st.session_state["messages"]
    # First call to GPT. Generates a draft response that will not be visible to the user.
    try:
        gpt_draft_response = client.chat.completions.create(
            model=deployment_name,
            messages=prompt_with_history,
            response_format=get_response_format(require_term_id=False)
        )
        gpt_draft_content_response = gpt_draft_response.choices[0].message.content
    except Exception as e:
        st.error(f"Error in initial GPT call: {e}")
        gpt_draft_content_response = ""

    print(gpt_draft_content_response)
    
    # Convert response to dict 
    if type(gpt_draft_content_response) == str:
        gpt_draft_content_response_dict = json.loads(gpt_draft_content_response)
        # Ensure term_id is present in the response
        if 'term_id' not in gpt_draft_content_response_dict:
            gpt_draft_content_response_dict['term_id'] = None

        print("Type: ", type(gpt_draft_content_response_dict))
        print("Content: ", gpt_draft_content_response_dict)
    
    # OLD Analyze and modify output
    # analyzed_output = analyze_initial_output(initial_content)

    idm = load_id_manual("./data/idmanual.csv", "./data/classes.csv")

    # Analyze the gpt draft response (For now: just Create search results)
    search_results = create_search_results(gpt_draft_content_response_dict, idm)

    # Create refinement prompt. This includes the search results
    refinement_prompt_content = create_refinement_prompt(search_results)
    refinement_prompt_with_history = prompt_with_history + [
        {
            "role": "assistant",
            "content": gpt_draft_content_response
        },
        {
            "role": "system",
            "content": refinement_prompt_content
        }
    ]
 
    # Second GPT call
    try:
        gpt_final_response = client.chat.completions.create(
            model=deployment_name,
            messages=refinement_prompt_with_history,
            response_format=get_response_format(require_term_id=True)
        )
        gpt_final_content_response = gpt_final_response.choices[0].message.content
    except Exception as e:
        st.error(f"Error in gpt_final_content_response call: {e}")
        gpt_final_content_response = "Error generating response."

    # Convert GPT output to dict
    if type(gpt_final_content_response) == str:
        gpt_final_content_response_dict = json.loads(gpt_final_content_response)
        assert type(gpt_final_content_response_dict) == dict


    # Display the free-form part of GPT final response to the user
    with st.chat_message("assistant"):
        st.markdown(gpt_final_content_response_dict["free_form_response"])

    # Append final GPT response to history. Note that the initial assistant message is NOT included.
    st.session_state["messages"].append(
        {"role": "assistant", "content": gpt_final_content_response_dict["free_form_response"]}
    )

    # Save conversation to CSV
    save_messages_to_csv(
        messages=st.session_state["messages"],
        session_key=st.session_state["session_key"]
    )


#QA individual functions here. Execute the code below when I run the script on this page
'''
if __name__ == "__main__":

    # load_id_manual
    idmanual_path = "./data/idmanual.csv"
    classes_path = "./data/classes.csv"
    idm = load_id_manual(idmanual_path, classes_path)
    print(len(idm))
    print(idm.head(3))
    print('')

    # search_one_term
    search_term = "medical device for cancer detection"
    class_id = "10"
    sort_by = "cosine_sim"
    max_nbr_terms_returned = 3
    search_result = search_one_term(search_term, class_id, idm, sort_by, max_nbr_terms_returned)
    pretty_search_result = json.dumps(search_result, indent=4)
    print(pretty_search_result)
    print('')

    # create_search_results
    gpt_draft_content = {
        "terms": [
            {"term": "medical device for cancer detection", "class_id": 10},
            {"term": "software for financial transactions", "class_id": 42}
        ]
    }
    max_nbr_terms_returned = 3
    search_results = create_search_results(gpt_draft_content, idm, max_nbr_terms_returned=max_nbr_terms_returned)
    pretty_search_results = json.dumps(search_results, indent=4)
    print(pretty_search_results)
    print('')


    # create_refinement_prompt
    refinement_prompt_content = create_refinement_prompt(search_results)
    print(refinement_prompt_content)
    print('')

    # save_messages_to_csv
    messages = [
        {"role": "user", "content": "I want to trademark a medical device for cancer detection."},
        {"role": "assistant", "content": "Consider Class 10 for medical devices."}
    ]
    session_key = "1234"
    save_messages_to_csv(messages, session_key)
    print("Messages saved to chat_history.csv")
    chat_history = pd.read_csv("chat_history.csv")
    print(chat_history)
'''