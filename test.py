from transformers import T5Tokenizer, T5ForConditionalGeneration, get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_generation(model, test_df, train_df, item_to_semantic, semantic_to_item, tokenizer, device, k=0):
    model.eval()
    results = []

    history_map = train_df.sort_values("time").groupby("user_id")["item_id"].apply(list).to_dict()

    for _, row in test_df.iterrows():
        user_id = row["user_id"]
        target_asin = row["parent_asin"]

        user_history = history_map.get(user_id, [])
        input_str = " ".join([item_to_semantic[item] for item in user_history if item in item_to_semantic])

        inputs = tokenizer(
            input_str,
            max_length=512,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        ).to(device)

        outputs = model.generate(
            input_ids=inputs["input_ids"],
            max_length=10,
            num_beams=k,
            num_return_sequences=k,
            early_stopping=True,
        )

        predictions = [tokenizer.decode(output, skip_special_tokens=False) for output in outputs]

        is_hit = any(semantic_to_item.get(pred.strip(), None) == target_asin for pred in predictions)
        results.append(is_hit)
    
    print(f"Hit Rate@{k if k > 0 else 1}: {sum(results) / len(results):.4f}")

if __name__ == "__main__":
    model_name = "google-t5/t5-base"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Add codeword tokens to tokenizer
    codebook_size = 256
    new_tokens = []
    for i in range(codebook_size):
        new_tokens.append(f"<codeword1_{i}>")
        new_tokens.append(f"<codeword2_{i}>")
        new_tokens.append(f"<codeword3_{i}>")

    # Add extra unique tokens to tokenizer
    for i in range(1000, 1100):
        new_tokens.append(f"<unique_identity_token_{i}>")

    new_added_tokens = tokenizer.add_tokens(new_tokens)
    model.resize_token_embeddings(len(tokenizer))
    model.to(device)

    # load user history data
    train_data = pd.read_csv("dataset/train_data.csv")
    eval_data = pd.read_csv("dataset/eval_data.csv")
    test_data = pd.read_csv("dataset/test_data.csv")

    mapping_df = pd.read_csv("dataset/semantic_id_to_item_id.csv")
    item_to_semantic_ids = {}
    for _, row in mapping_df.iterrows():
        semantic_id = f"<codeword1_{row["codeword1"]}> <codeword2_{row["codeword2"]}> <codeword3_{row["codeword3"]}> <unique_identity_token_{row["unique_identity_token"]}>"
        item_to_semantic_ids[row["item_id"]] = semantic_id

    semantic_ids_to_item = {v: k for k, v in item_to_semantic_ids.items()}

    evaluate_generation(
        model,
        test_data,
        train_data,
        item_to_semantic_ids,
        semantic_ids_to_item,
        tokenizer,
        device=device,
        k=5,
    )