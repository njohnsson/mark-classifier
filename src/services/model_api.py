# src/services/model_api.py
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))  # Up 2 steps

import json
from openai import AzureOpenAI
from src.config.config import Config
from src.config.constants import LLM_MODEL_DEPLOYMENT_NAME, LLM_MODEL_API_VERSION, EMBEDDING_MODEL_DEPLOYMENT_NAME

def initialize_openai_client():
    """ Initialize an Azure Open AI client. """

    openai_client = AzureOpenAI(
        api_key=Config.AZURE_API_KEY,
        azure_endpoint=Config.AZURE_ENDPOINT,
        api_version=LLM_MODEL_API_VERSION
    )
    return openai_client



def get_embedding(openai_client, term):
    """Get the embedding of a term (phrase). Output list of floats."""
    
    embedding = openai_client.embeddings.create(input=[term], model=EMBEDDING_MODEL_DEPLOYMENT_NAME).data[0].embedding
    
    return embedding



def get_gpt_response(client, messages, response_format):
    """Get a response from the GPT model. Outputs response content and the full response object."""
    
    response = client.chat.completions.create(
        model=LLM_MODEL_DEPLOYMENT_NAME,
        messages=messages,
        response_format=response_format
    )

    content_response = response.choices[0].message.content

    # If response_format is set, output should be JSON. Need to parse it.
    if response_format:
        try:
            content_response = json.loads(content_response)
        except json.JSONDecodeError:
            print("Error parsing JSON response.")
        
    return content_response, response



# Define the Open AI API response format
def get_response_format(require_term_id=False):
    """
    Define the JSON schema for the OpenAI API response.

    Parameters:
        require_term_id (bool): Whether the response should include a term ID.

    Returns:
        dict: The response format.
    """

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



# QA individual functions here.
if __name__ == "__main__":
    
    openai_client = initialize_openai_client()
    print('openai_client: ', openai_client)
    print('')

    print("get_response_format(require_term_id=False)")
    response_format = get_response_format(require_term_id=False)
    print(json.dumps(response_format, indent=4))
    print('')
    
    print("get_response_format(require_term_id=True)")
    response_format = get_response_format(require_term_id=True)
    print(json.dumps(response_format, indent=4))
    print('')
    print('')

    print("get_embedding(openai_client, 'silly labrador')")
    emb = get_embedding(openai_client, "silly labrador")
    print('Type emb: ', type(emb))
    print('Len Emb: ', len(emb))
    print(emb[:3])
    print('')
    
    print("get_gpt_response, with response_format=False")

    messages = [
    {
        "role": "user",
        "content": "I have developed a harness for bears."
    },
    {
        "role": "assistant",
        "content": (
            "This could fall into **Class 18 - Leather goods**, especially the term "
            "_harnesses for animals_. It may also be relevant to **Class 28 - Toys "
            "and sporting goods**. Please provide some more details. What is the "
            "material? In what situations would the harness be used?"
        )
    },
    {
        "role": "user",
        "content": (
            "The harness is made of leather and nylon. It will be used by researchers "
            "studying bear behavior. The harness lets researchers mount a camera on "
            "the bear as well as GPS tracking devices."
        )
    }
    ]

    
    content_response, _ = get_gpt_response(openai_client, messages, get_response_format(require_term_id=False))
    print(json.dumps(content_response, indent=4))
    print('')
    
    # TO DO
    # print("get_gpt_response, with response_format=True")
    # print(get_gpt_response(openai_client, ["message1", "message2"], get_response_format(require_term_id=True)))