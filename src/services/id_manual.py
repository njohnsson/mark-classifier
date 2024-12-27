# src/services/load_id_manual.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import pandas as pd
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


