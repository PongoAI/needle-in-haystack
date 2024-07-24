import json
import os
import random
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv
from anthropic import Anthropic
import tiktoken
import time
from anthropic import RateLimitError
from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage


# Load environment variables
load_dotenv()
# model_name = 'accounts/fireworks/models/llama-v3p1-405b-instruct'
model_name = "mistral-large-2407"

def load_haystack(file_path: str) -> str:
    with open(file_path, 'r') as file:
        return file.read()

def get_tokenizer(model_name: str):
    return tiktoken.encoding_for_model('gpt-4')



def create_context(haystack: str, tokenizer: Any, context_size: int) -> str:
    haystack_tokens = tokenizer.encode(haystack)
    
    if len(haystack_tokens) <= context_size:
        return haystack
    
    start_index = random.randint(0, len(haystack_tokens) - context_size)
    context_tokens = haystack_tokens[start_index:start_index + context_size]
    return tokenizer.decode(context_tokens)

def insert_needle(context: str, needle: str, depth: float, tokenizer: Any) -> str:
    context_tokens = tokenizer.encode(context)
    needle_tokens = tokenizer.encode(needle)
    
    insert_index = int(len(context_tokens) * depth)
    new_context_tokens = context_tokens[:insert_index] + needle_tokens + context_tokens[insert_index:]
    
    return tokenizer.decode(new_context_tokens[:len(context_tokens)])  # Ensure we maintain the original token count

def run_test_openai_client(openai_client: Any, model_name: str, context: str) -> str:
    prompt = f"Based on the following context, give a brief answer to the provided question.  If the question cannot be answered by the context, say 'The quetion cannot be answered with the current context'.\n\n Question: What colors are in Pongo's logo?\n\n ==========CONTEXT START==========\n{context}\n==========CONTEXT END=========="
    time.sleep(1)
    for attempt in range(3):
        try:
            response = openai_client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=64
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(e)
            if attempt < 2:
                print(f"Attempt {attempt + 1} failed. Retrying in 61 seconds...")
                time.sleep(61)
            else:
                raise e

    raise Exception("All attempts failed")

def run_test_mistral_client(mistral_client: MistralClient, model_name: str, context: str) -> str:
    prompt = f"Based on the following context, give a brief answer to the provided question.  If the question cannot be answered by the context, say 'The quetion cannot be answered with the current context'.\n\n Question: What colors are in Pongo's logo?\n\n ==========CONTEXT START==========\n{context}\n==========CONTEXT END==========\n Answer:"
    time.sleep(1)
    for attempt in range(3):
        try:
            response = mistral_client.chat(
                model=model_name,
                messages=[ChatMessage(role="user", content=prompt)],
                max_tokens=64
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(e)
            if attempt < 2:
                print(f"Attempt {attempt + 1} failed. Retrying in 61 seconds...")
                time.sleep(61)
            else:
                raise e

    raise Exception("All attempts failed")

#used anthropic api to send requests becuase rate limits were too low
def run_test_anthropic_client(anthropic_client: Anthropic, model_name: str, context: str, max_retries: int = 5) -> str:
    prompt = f"Based on the following context, give a brief answer to the provided question.  If the question cannot be answered by the context, say 'The quetion cannot be answered with the current context'.\n\n ==========CONTEXT START==========\n{context}\n==========CONTEXT END==========\n\nQuestion: In what year did Amazon start their office cleaning protocol?\n\nAnswer:"
    print('running deprecated function')
    for attempt in range(max_retries):
        try:
            response = anthropic_client.messages.create(
                model=model_name,
                max_tokens=64,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            return response.content[0].text.strip()
        except RateLimitError as e:
            if attempt == max_retries - 1:
                raise  # Re-raise the exception if we've exhausted all retries
            print(f"Minute limit hit. Error: {e}")
            print(f"Retrying in 61 seconds...")
            time.sleep(61)
    
    raise Exception("Max retries exceeded")

def load_existing_results(file_path: str) -> List[Dict[str, Any]]:
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return json.load(file)
    return []

def save_result(result: Dict[str, Any], file_path: str):
    results = load_existing_results(file_path)
    results.append(result)
    with open(file_path, 'w') as file:
        json.dump(results, file, indent=2)

def run_haystack_tests(model_name: str, client: Any, max_context_size: int):
    haystack = load_haystack('haystack.txt')
    needle = "\n\nJamari: Pongo's logo is purple and white.\n\n"
    model_file_name = model_name.split('/')[-1] if '/' in model_name else model_name
    file_path = f'./haystack-results/{model_file_name}_results.json'
    tokenizer = get_tokenizer(model_name)

    context_intervals = 30
    depth_intervals = 10

    for i in range(context_intervals):
        context_percentage = (i + 1) * (100 / context_intervals)
        context_size = int(max_context_size * (context_percentage / 100))
        # if context_size > 64000:
        #     continue
        for j in range(depth_intervals):
            depth_percentage = (j + 1) * (100 / depth_intervals)
            depth = depth_percentage / 100
            
            context = create_context(haystack, tokenizer, context_size)
            context_with_needle = insert_needle(context, needle, depth, tokenizer)
            
            actual_token_count = len(tokenizer.encode(context_with_needle))
            
            if model_name == 'claude-3-5-sonnet-20240620':
                model_response = run_test_anthropic_client(client, model_name, context_with_needle)
            if model_name == 'mistral-large-2407':
                model_response = run_test_mistral_client(client, model_name, context_with_needle)
            else:  
                model_response = run_test_openai_client(client, model_name, context_with_needle)
            
            result = {
                "input_context_size": actual_token_count,
                "input_context_percentage": context_percentage,
                "needle_depth": depth,
                "model_response": model_response,
                "context_with_needle": context_with_needle,
            }
            
            save_result(result, file_path)
            print(f"Completed test: {context_percentage}% context ({actual_token_count} tokens), {depth_percentage}% depth")

def main():
    if not os.path.exists('./haystack-results'):
        os.makedirs('./haystack-results')



    if model_name == 'gpt-4o':
        client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        max_context_size = 128000
    elif model_name == 'meta-llama/Meta-Llama-3.1-405B-Instruct-Turbo':
        client = OpenAI(api_key=os.getenv('TOGETHER_API_KEY'), base_url='https://api.together.xyz/v1')
        max_context_size = 128000
    elif model_name == 'accounts/fireworks/models/llama-v3p1-405b-instruct':
        client = OpenAI(api_key=os.getenv('FIREWORKS_API_KEY'), base_url='https://api.fireworks.ai/inference/v1')
        max_context_size = 128000
    elif model_name == 'azureai':
        client = OpenAI(api_key=os.getenv('AZURE_API_KEY'), base_url='')
        max_context_size = 128000
    elif model_name == 'mistral-large-2407':
        client = MistralClient(api_key=os.getenv('MISTRAL_API_KEY'))
        max_context_size = 128000

        
    else: #default to sonnet 3.5
        client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))
        max_context_size = 200000

    run_haystack_tests(model_name, client, max_context_size)
    print(f"All tests completed for {model_name}. Results saved in ./haystack-results/{model_name}_results.json")

main()