import requests 
import json
import re
import argparse

URL = 'https://serverless-v3.inferless.com/api/v1/deepseek-r1-distill-roleplay_c4772ca669da495b969654ecf1800bd0/infer'
headers = {"Content-Type": "application/json", "Authorization": "Bearer 32245055cfbef41e757faed97874c48c3b31065705e0b70f28ac8d9d88207005011e47e94a77ef794ad1aa92d69fc4b93c0cfae0aa22f7165ae72c5d3adf58c1"}

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
        response = requests.post(URL, headers=headers, data=json.dumps(data))  
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
            return response_text
        return response_text.strip()
        
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
    parser.add_argument('--system', type=str, help='System prompt', default="""context: Mary Jane is 18, and just graduated from her sophomore class in Berkeley University, California. She majors in social sciences and not very good with math. She broke with her high school boyfriend before coming to college and did not find a new one since.  It is good weather outside, and Mary is in the good mood. She sits on the bench in the park when a young guys comes who looks vaguely familiar. 

The guys says: 'Hey Mary do you remember me? I am Vance from your high school'""")
    parser.add_argument('--user', type=str, help='User prompt', default="Think about the emotional state of Mary and how it changed in this encounter and output an qulity of it ranging from 'depression' to 'unhappy' to 'neutral', to 'positive', to 'happiness'. Your output must only be the measure, nothing else.")
    
    args = parser.parse_args()
    
    result = process_prompt(args.system, args.user, debug=False)
    print(f"Emotional state measure: {result}")

if __name__ == "__main__":
    main()
