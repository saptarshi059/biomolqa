import sys
sys.path.append("../common_scripts/")

from common_functions import save_batch, print_sample, count_tokens, create_formatted_samples_for_eval
from eval_prompts import SYSTEM_PROMPT_ZERO_SHOT, USER_PROMPT_ZERO_SHOT
import json
import pandas as pd
test_df = pd.read_csv("../../dataset_for_hf/test.csv")

def run_deepseek_inference(input_path, output_path):
    responses = []
    with open(input_path, 'r') as f:
        for line in f:
            sample = json.loads(line.strip())
            messages = sample["body"]["messages"]
            model = sample["body"]["model"]
            
            try:
                response = client.chat.completions.create(
                    model=model,
                    messages=messages,
                    stream=False
                )
                output = {
                    "custom_id": sample["custom_id"],
                    "response": response.choices[0].message.content
                }
            except Exception as e:
                output = {
                    "custom_id": sample["custom_id"],
                    "response": f"[ERROR] {str(e)}"
                }

            responses.append(output)

    # Save responses
    with open(output_path, 'w') as f:
        for r in responses:
            f.write(json.dumps(r) + '\n')

    print(f"Saved {len(responses)} results to {output_path}")
    
def create_formatted_inputs_for_upper_bound_eval(row):
    return {
        "custom_id": f"{row.Entities}_{row.Label}:upper_bound",
        "body": {
            "model": 'deepseek-reasoner',
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_ZERO_SHOT},
                {"role": "user", "content": f"QUESTION: {row.Question}\n\nRELEVANT KNOWLEDGE: {row.Question_Background}"}
            ]
        }
    }

# Apply to test_df
formatted_upper_bound = [create_formatted_inputs_for_upper_bound_eval(row) for row in test_df.itertuples()]
save_batch(formatted_upper_bound, "../../samples_for_eval/upper_bound/deepseek/batch_input.jsonl")

from openai import OpenAI

client = OpenAI(api_key="", base_url="https://api.deepseek.com")
run_deepseek_inference('../../samples_for_eval/upper_bound/deepseek/batch_input.jsonl', "../../samples_for_eval/upper_bound/deepseek/responses_r1.jsonl")