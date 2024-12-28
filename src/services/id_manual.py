# src/services/load_id_manual.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import pandas as pd
import Levenshtein
from sklearn.metrics.pairwise import cosine_similarity
from src.services.model_api import get_embedding
from src.config.constants import RAW_IDMANUAL_CSV_PATH, CLASSES_CSV_PATH, IDM_CSV_PATH

def create_idm(idmanual_path=RAW_IDMANUAL_CSV_PATH, classes_path=CLASSES_CSV_PATH, idm_csv_path=IDM_CSV_PATH):
    """
    Load the raw ID manual, clean up, and enrich it with class names from a lookup file.
    
    Parameters:
        idmanual_path (str): Path to the ID manual CSV file.
        classes_path (str): Path to the classes lookup CSV file.
        idm_csv_path (str): Path to save the cleaned ID manual CSV file.
    
    Returns:
        idm (pd.DataFrame): A cleaned and enriched DataFrame of the ID manual.
    """
    # Load ID manual. Class is a string.
    idm_raw = pd.read_csv(idmanual_path, low_memory=False)
    
    # Load classes lookup. class_id will be an int.
    classes_df = pd.read_csv(classes_path)
    classes_df['class_id'] = classes_df['class_id'].astype(str)
    
    # Merge class names (on class id string) into the ID manual
    idm = idm_raw.merge(classes_df, left_on="Class", right_on="class_id", how='left')
    idm.drop(columns=['class_id'], inplace=True)
    
    # Clean up column names
    idm.rename(columns={'Class' : 'class_id'}, inplace=True)
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

    # Save the cleaned ID manual to disk
    idm.to_csv(idm_csv_path, index=False)
    
    print(f"Loaded {len(idm)} records from ID manual.")

def load_idm(idm_csv_path=IDM_CSV_PATH):
    """
    Load the cleaned ID manual.
    
    Returns:
        idm (pd.DataFrame): A cleaned and enriched DataFrame of the ID manual.
    """
    idm = pd.read_csv(idm_csv_path)

    # Restore data types
    idm.effective_date = pd.to_datetime(idm.effective_date, errors='coerce')
    idm['class_id'] = idm['class_id'].astype(str) # # From object to string
    
    return idm

def search_one_term(search_term, class_id, idm, openai_client, sort_by="cosine_sim", max_nbr_terms_returned=10):
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
    search_term_emb = get_embedding(openai_client, search_term)

    # filter the idm by class_id
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

def create_search_results(gpt_draft_content, idm, openai_client, max_nbr_terms_returned=10):
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
        result = search_one_term(term, class_id, idm, openai_client, max_nbr_terms_returned=max_nbr_terms_returned)
        search_results.append(result["search_results"][0])
    return search_results




# QA
if __name__ == '__main__':
    
    from src.config.constants import TEMP_DIR

    # Test create_idm. Put file in temp directory
    idm_csv_path = os.path.join(TEMP_DIR, 'idm_test.csv')
    create_idm(idmanual_path=RAW_IDMANUAL_CSV_PATH, classes_path=CLASSES_CSV_PATH, idm_csv_path=idm_csv_path)

    # Test load_idm 
    idm = load_idm(idm_csv_path=idm_csv_path)
    print(idm.head())
    print(idm.shape)
    print(idm.dtypes)


