from transformers import T5Tokenizer, T5ForConditionalGeneration, get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_generation(model, test_df, train_df, item_to_semantic, semantic_to_item, tokenizer, device, k=0):
    model.eval()
    results = []

    history_map = train_df.sort_values("time").groupby("user_id")["parent_asin"].apply(list).to_dict()

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
            attention_mask=inputs["attention_mask"],
            max_length=10,
            num_beams=k,
            num_return_sequences=k,
            decoder_start_token_id=tokenizer.pad_token_id,
            early_stopping=True,
        )

        predictions = [tokenizer.decode(output, skip_special_tokens=False) for output in outputs]

        print(f"User ID: {user_id}, Target ASIN: {target_asin}, Predictions: {predictions}")

        break

        is_hit = any(semantic_to_item.get(pred.strip(), None) == target_asin for pred in predictions)
        results.append(is_hit)
    
    print(f"Hit Rate@{k if k > 0 else 1}: {sum(results) / len(results):.4f}")

if __name__ == "__main__":
    model_name = "google-t5/t5-base"
    tokenizer = T5Tokenizer.from_pretrained("checkpoints/tiger_model_epoch_6")
    model = T5ForConditionalGeneration.from_pretrained("checkpoints/tiger_model_epoch_6")
    model.to(device)

    # 1. 检查你的特殊 Token 是否真的在词表里
    test_token = "<unique_identity_token_1001>"
    token_id = tokenizer.convert_tokens_to_ids(test_token)
    print(f"Token: {test_token}, ID: {token_id}")

    # 2. 检查 Tokenizer 是如何切分你的输入字符串的
    test_str = "<codeword1_1> <unique_identity_token_1001>"
    encoded = tokenizer.encode(test_str, add_special_tokens=False)
    decoded_steps = [tokenizer.decode([tid]) for tid in encoded]
    print(f"Tokenization steps: {decoded_steps}")

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
        eval_data,
        train_data,
        item_to_semantic_ids,
        semantic_ids_to_item,
        tokenizer,
        device=device,
        k=5,
    )