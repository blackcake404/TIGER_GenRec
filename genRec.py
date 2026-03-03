from transformers import T5Tokenizer, T5ForConditionalGeneration, get_linear_schedule_with_warmup
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from torch.optim import AdamW
from tqdm import tqdm

def sliding_window_samples(train_df, item_to_semantic_ids, max_history_length=20):
    train_df = train_df.sort_values(by=["user_id", "time"])
    user_groups = train_df.groupby("user_id")["parent_asin"].apply(list)

    samples = []

    for user_id, item_list in user_groups.items():
        if len(item_list) < 2:
            continue

        semantic_id_list = [item_to_semantic_ids[parent_asin] for parent_asin in item_list if parent_asin in item_to_semantic_ids]

        for i in range(1, len(semantic_id_list)):
            start_index = max(0, i - max_history_length)
            history_subset = semantic_id_list[start_index:i]

            input_str = " ".join(history_subset) # input for encoder
            target_str = semantic_id_list[i] # target for decoder

            samples.append({
                "user_id": user_id,
                "input_text": input_str,
                "target_text": target_str
            })
    
    return pd.DataFrame(samples)

class TIGERDataset(Dataset):
    def __init__(self, data_df, tokenizer, max_len=512):
        self.data = data_df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        inputs = self.tokenizer(
            row["input_text"],
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        target = self.tokenizer(
            row["target_text"],
            max_length=10,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = target.input_ids.squeeze()
        labels[labels == self.tokenizer.pad_token_id] = -100  # ignore padding in loss

        return {
            "input_ids": inputs.input_ids.squeeze(),
            "attention_mask": inputs.attention_mask.squeeze(),
            "labels": labels,
        }
        
def tiger_dataloader(data_df, tokenizer, batch_size=32, shuffle=False):
    dataset = TIGERDataset(
        data_df=data_df,
        tokenizer=tokenizer,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True,
    )

def train_tiger_model(model, train_dataloader, optimizer, scheduler, device, num_epochs):
    model.train()

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}")

        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attenttion_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attenttion_mask,
                labels=labels,
            )

            loss = outputs.loss
            epoch_loss += loss.item()

            loss.backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            progress_bar.set_postfix({"Loss": f"{loss.item():.4f}"})
        
        avg_epoch_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} completed. Average Loss: {avg_epoch_loss:.4f}")

        model.save_pretrained(f"checkpoints/tiger_model_epoch_{epoch+1}")

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
    print(f"Added {new_added_tokens} new tokens to the tokenizer.")
    model.resize_token_embeddings(len(tokenizer))

    # load user history data
    train_data = pd.read_csv("dataset/train_data.csv")
    eval_data = pd.read_csv("dataset/eval_data.csv")
    test_data = pd.read_csv("dataset/test_data.csv")

    # mapping from parent_asin to semantic ids
    mapping_df = pd.read_csv("dataset/semantic_id_to_item_id.csv")
    item_to_semantic_ids = {}
    for _, row in mapping_df.iterrows():
        semantic_id = f"<codeword1_{row["codeword1"]}> <codeword2_{row["codeword2"]}> <codeword3_{row["codeword3"]}> <unique_identity_token_{row["unique_identity_token"]}>"
        item_to_semantic_ids[row["item_id"]] = semantic_id

    train_samples_df = sliding_window_samples(train_df=train_data, item_to_semantic_ids=item_to_semantic_ids)

    print(f"Train smaple df shape: {train_samples_df.shape}")

    train_dataloader = tiger_dataloader(
        data_df=train_samples_df,
        tokenizer=tokenizer,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

    num_epochs = 10
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps,
    )

    train_tiger_model(
        model=model,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=num_epochs,
    )