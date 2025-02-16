import requests 
import json
import os
import dotenv
import argparse
from pydantic import BaseModel, Field
from typing import List
from openai import OpenAI

dotenv.load_dotenv(override=True)

OPENAI_API_KEY = ""
URL = 'https://serverless-v3.inferless.com/api/v1/deepseek-r1-distill-roleplay_c4772ca669da495b969654ecf1800bd0/infer'
headers = {"Content-Type": "application/json", "Authorization": "Bearer "}


class EmotionalMeter(BaseModel):
    current_mood: int = Field(
        ...,
        description=f"Current emotional state change as integer in range -50 (lowest) to +50 (highest) in the increments of 10."
    )


def get_structured_info_from_text(text: str, query_class: BaseModel) -> BaseModel:
    """
    Extract structured information from the provided text using OpenAI's chat API.
    """
    def get_extraction_messages(text: str, query_class: BaseModel):
        return [
            {"role": "system", "content": "You are an AI assistant skilled in extracting changes inemotional state from context of human life."},
            {"role": "user", "content": (
                f"Extract structured information based on the following JSON schema: "
                f"{json.dumps(query_class.model_json_schema()['properties'], indent=2)}. "
                f"Context to extract from: {text}"
            )}
        ]
    messages = get_extraction_messages(text, query_class)

    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    response = openai_client.chat.completions.create(
        model="gpt-4o-2024-08-06",
        messages=messages,
    )
    if response and response.choices and response.choices[0].message.content:
        extracted_json = response.choices[0].message.content.strip()
        if extracted_json:
            response2 = openai_client.beta.chat.completions.parse(
                model="gpt-4o-2024-08-06",
                messages=[
                    {"role": "system", "content": "Extract proper JSON from the following text."},
                    {"role": "user", "content": extracted_json}
                ],
                response_format=query_class,
            )
            if response2 and response2.choices and response2.choices[0].message.parsed:
                return response2.choices[0].message.parsed
            else:
                print("Failed to parse structured info.")
                return query_class()
        else:
            print("Empty response from OpenAI.")
            return query_class()
    else:
        print("Failed to retrieve structured info.")
        return query_class()



def process_prompt(system_prompt, user_prompt, debug=False):
    
    data = {
        "inputs": [
            {
                "name": "system_prompt",
                "shape": [1],
                "data": [system_prompt],
                "datatype": "BYTES"
            },
            {
                "name": "user_prompt",
                "shape": [1],
                "data": [user_prompt],
                "datatype": "BYTES"
            },
            {
                "name": "temperature",
                "optional": True,
                "shape": [1],
                "data": [0.7],
                "datatype": "FP32"
            },
            {
                "name": "top_p",
                "optional": True,
                "shape": [1],
                "data": [0.1],
                "datatype": "FP32"
            },
            {
                "name": "repetition_penalty",
                "optional": True,
                "shape": [1],
                "data": [1.18],
                "datatype": "FP32"
            },
            {
                "name": "max_tokens",
                "optional": True,
                "shape": [1],
                "data": [1024],
                "datatype": "INT16"
            },
            {
                "name": "top_k",
                "optional": True,
                "shape": [1],
                "data": [40],
                "datatype": "INT8"
            }
        ]
    }
    
    try:
        response = requests.post(URL, headers=headers, data=json.dumps(data), timeout=10)
        if debug:
            print(f"Status Code: {response.status_code}")
            print(f"Response Text: {response.text}")
            
        if response.status_code != 200:
            print(f"Error: API returned status code {response.status_code}")
            print(f"Response: {response.text}")
            return "error"
            
        response_json = response.json()
        
        # Process the response
        response_text = response_json.get('outputs', [{'data': ['']}])[0]['data'][0]
        
        # Cut everything up to and including </think>
        if '</think>' in response_text:
            response_text = response_text.split('</think>', 1)[1].strip()
        return get_structured_info_from_text(response_text.strip(), EmotionalMeter)
        
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return "error"
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        print(f"Response content: {response.text}")
        return "error"
    except Exception as e:
        print(f"Unexpected error: {e}")
        return "error"

def main():
    parser = argparse.ArgumentParser(description='Process prompts with Sherlock Holmes AI')
    parser.add_argument('--system', type=str, help='System prompt', default="""context: Mary Jane is 18, and just graduated from her sophomore class in Berkeley University, California. She majors in social sciences and is not very good with math. She broke up with her high school boyfriend before coming to college and did not find a new one since. It is good weather outside, and Mary is in a good mood. She sits on the bench in the park when a young guy comes who looks vaguely familiar.\n\nThe guy says: 'Hey Mary, do you remember me? I am Vance from your high school.'""")
    parser.add_argument('--user', type=str, help='User prompt', default="Think about the emotional state of the user and estimate how it changed in this encounter.")
    
    args = parser.parse_args()
    
    result = process_prompt(args.system, args.user, debug=True)
    print(f"Emotional state measure: {result}")

if __name__ == "__main__":
    main()
