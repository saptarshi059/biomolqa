def print_sample(sample_list):
    sample = sample_list[0]
    print(sample["body"]["messages"][0]["content"] + "\n\n" + sample["body"]["messages"][1]["content"])

def count_tokens(sample_list):
    import tiktoken
    gpt_tokenizer = tiktoken.encoding_for_model("gpt-4o")
    
    total_tokens = 0
    for sample in sample_list:
        total_tokens = total_tokens + len(gpt_tokenizer.encode(sample["body"]["messages"][0]["content"])) + \
                                      len(gpt_tokenizer.encode(sample["body"]["messages"][1]["content"]))
    print(total_tokens)

def retrieve_text(entity_name, entity_type):
    complexified_path = Path(f"../data/background_information_data/{entity_type}_data/Wiki_complexified/{entity_name}.txt")
    if complexified_path.exists():
        with complexified_path.open("r") as file:
            return file.read()
    else:
        with Path(f"../data/background_information_data/{entity_type}_data/Wiki/{entity_name}.txt").open("r") as file:
            return file.read()

def save_batch(batch, output_path):
    from pathlib import Path
    import json
    
    with Path(output_path).open('w') as file:
        for sample in batch:
            json_line = json.dumps(sample)
            file.write(json_line + '\n')

def create_batches(formatted_samples, base_path):
    batch_tokens = 0
    batch_id = 0
    batch = []
    i = 0
    while i < len(formatted_samples):
        sample = formatted_samples[i]
        batch.append(sample)
        batch_tokens = batch_tokens + \
                       len(gpt_tokenizer.encode(sample["body"]["messages"][0]["content"])) + \
                       len(gpt_tokenizer.encode(sample["body"]["messages"][1]["content"]))
        if batch_tokens > 90_000:
            batch.pop() # Removing the last sample which caused the total number of tokens to exceed the 90K limit.
            save_batch(batch, f"{base_path}/batch_{batch_id}_input.jsonl")
            batch_id += 1
            batch = []
            batch_tokens = 0
        else:
            i += 1
    save_batch(batch, f"{base_path}/batch_{batch_id}_input.jsonl")

def create_formatted_samples_for_eval(df, output_path, model_type):
    from pathlib import Path
    import unicodedata
    import pickle
    import re

    predictions_for_squad = []
    references_for_squad = []
    predictions_for_bertscore = []
    references_for_bertscore = []
    for row in df.itertuples():
        gold = re.escape(row.Answer)

        if model_type == "OAI":
            pred = unicodedata.normalize('NFKD', re.search("(ANSWER:)(.*)", row.response["body"]["choices"][0]["message"]["content"]).group(2).strip())
            predictions_for_squad.append({'prediction_text': pred, 'id': row.id})
            references_for_squad.append({'answers': {'answer_start': [0], 'text': [gold]}, 'id': row.id})
        elif model_type == "Anthropic":
            pred = unicodedata.normalize('NFKD', re.search("(ANSWER:)(.*)", row.result["message"]["content"][0]["text"]).group(2).strip())
            predictions_for_squad.append({'prediction_text': pred, 'id': row.Entities})
            references_for_squad.append({'answers': {'answer_start': [0], 'text': [gold]}, 'id': row.Entities})

        predictions_for_bertscore.append(pred)
        references_for_bertscore.append(gold)

    with Path(output_path).open("wb") as file:
        pickle.dump((predictions_for_squad, references_for_squad, predictions_for_bertscore, references_for_bertscore), file)