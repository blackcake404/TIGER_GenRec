import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict

BETA = 0.25

EPOCH = 15
LR = 5e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OFFSET = 1000

# generate item embeddings

SENTENCE_TEMPLATE = """
    Product infomation:
    Level1_category: {l1_category},
    Level2_category: {l2_category},
    Level3_category: {l3_category},
    Title: {title},
    Average rating: {average_rating},
    Rating number: {rating_number},
    Features: {features},
    Description: {description},
    Price: {price},
    details: {details},
"""

def gen_sentence(row):
    sentence = SENTENCE_TEMPLATE.format(
        l1_category=row["l1_category"],
        l2_category=row["l2_category"],
        l3_category=row["l3_category"],
        title=row["title"],
        average_rating=row["average_rating"],
        rating_number=row["rating_number"],
        features=row["features"],
        description=row["description"],
        price=row["price"],
        details=row["details"],
    )
    return sentence

item_df = pd.read_csv("dataset/item_df.csv")

sentences = item_df.apply(gen_sentence, axis=1).tolist()

model = SentenceTransformer("sentence-transformers/sentence-t5-base")
embeddings = model.encode(sentences, batch_size=32, show_progress_bar=True)

np.save("dataset/item_embeddings.npy", embeddings)

# RQ-VAE
# 三中间层 512 256 128
class RQ_VAE_Encoder(nn.Module):
    def __init__(self, input_dim=768, latent_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, x):
        return self.net(x)

class RQ_VAE_Quantizer(nn.Module):
    def __init__(self, layer_num=3, codebook_size=256, codeword_dim=32):
        super().__init__()
        self.layer_num = layer_num
        self.codebooks = nn.ModuleList([
            nn.Embedding(codebook_size, codeword_dim) for _ in range(layer_num) # 每个 codebook 为 (codebook size, codeword dim)
        ])
        for codebook in self.codebooks:
            nn.init.uniform_(codebook.weight, -1.0/codebook_size, 1.0/codebook_size)
        

    def forward(self, z): # z for embedding (batch size, latent dim)
        residual = z
        all_codeword_idx = []
        quantized_result = 0.0
        for idx in range(self.layer_num):
            # embedding (batch size, 1, dim) / codebook (1, codebook size, dim) -> (batch size, 1，codebook size)
            codebook = self.codebooks[idx]
            distance = torch.cdist(residual.unsqueeze(1), codebook.weight.unsqueeze(0)).squeeze(1) # (batch size, codebook size)
            codeword_idx = torch.argmin(distance, dim=1)
            all_codeword_idx.append(codeword_idx)
            
            codeword = codebook(codeword_idx) # 取码字
            residual = residual - codeword # 更新残差，用于下一层量化
            quantized_result = quantized_result + codeword # 累计量化结果，作为最终输出

        return quantized_result, all_codeword_idx # 返回最终的累积量化结果和所有码字的索引

class RQ_VAE_Decoder(nn.Module):
    def __init__(self, input_dim=32, output_dim=768):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, output_dim),
        )

    def forward(self, x):
        return self.net(x)

class RQ_VAE(nn.Module):
    def __init__(self, encoder, quantizer, decoder):
        super().__init__()
        self.encoder = encoder
        self.quantizer = quantizer
        self.decoder = decoder

    def forward(self, x):
        z = self.encoder(x)
        quantized_z, all_codeword_idx = self.quantizer(z)
        x_recon = self.decoder(quantized_z)

        return {
            "z_embedding": z,
            "recon_x": x_recon,
            "quantized_z": quantized_z,
            "all_codeword_idx": all_codeword_idx,
        }
    
def compute_loss(x, recon_x, residual, codeword):
    # 重建损失
    recon_loss = nn.functional.mse_loss(x, recon_x)
    # 量化损失，优化码本
    codebook_loss = nn.functional.mse_loss(residual.detach(), codeword)
    # 提交损失，优化编码器，使其输出更接近码字
    commitment_loss = nn.functional.mse_loss(residual, codeword.detach())
    return recon_loss + codebook_loss + BETA * commitment_loss

# Dataset and DataLoader
class ItemEmbeddingDataset(Dataset):
    def __init__(self):
        self.embeddings = np.load("dataset/item_embeddings.npy").astype(np.float32)

    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, index):
        return torch.from_numpy(self.embeddings[index])
    
class ItemEmbedddingDataLoader(DataLoader):
    def __init__(self, batch_size=32, shuffle=True):
        dataset = ItemEmbeddingDataset()
        super().__init__(
            dataset, 
            batch_size=batch_size, 
            shuffle=shuffle,
            drop_last=True,
            pin_memory=True,
            )
        
# Training loop
def train_rqvae(model, data_loader, epochs=EPOCH, lr=LR, device=DEVICE):
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    model.to(device)
    model.train()

    for epoch in range(epochs):
        train_loss = 0.0
        for batch in data_loader: # batch (batch size, embedding dim) 之前生成的 item embedding
            batch = batch.to(device)
            optimizer.zero_grad()

            output = model(batch)

            # 计算 loss
            loss = compute_loss(batch, output["recon_x"], output["z_embedding"], output["quantized_z"])

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(data_loader):.4f}")

model = RQ_VAE(
    encoder=RQ_VAE_Encoder(),
    quantizer=RQ_VAE_Quantizer(),
    decoder=RQ_VAE_Decoder(),
)

data_loader = ItemEmbedddingDataLoader()

train_rqvae(model, data_loader)

# Generate semantic IDs
model.eval()
with torch.no_grad():
    all_item_embeddings = torch.from_numpy(embeddings).to(DEVICE)
    results = model(all_item_embeddings.float())

    indices = results["all_codeword_idx"] # indeces 为一个 list，长度为层数 3，每个元素为 (number of items, )

    raw_semantic_ids = torch.stack(indices, dim=1).cpu().numpy() # (number of items, 3)

    unique_semantic_ids = []
    path_counter = defaultdict(int)

    for semantic_id in raw_semantic_ids:
        semantic_id_tuple = tuple(semantic_id)
        suffix = OFFSET + path_counter[semantic_id_tuple] # 从偏移量 1000 开始
        path_counter[semantic_id_tuple] += 1

        new_id = list(semantic_id) + [suffix]
        unique_semantic_ids.append(new_id)
    
    semantic_ids = np.array(unique_semantic_ids) # (number of items, 4)

np.save("dataset/semantic_ids.npy", semantic_ids)

# Create semantic ID to item ID mapping
semantic_id_to_item_id_df = pd.DataFrame(
    semantic_ids,
    columns = ["codeword1", "codeword2", "codeword3", "unique_identity_token"],
)
semantic_id_to_item_id_df.insert(0, "item_id", item_df["parent_asin"].values)

is_unique = not semantic_id_to_item_id_df.duplicated(subset=["codeword1", "codeword2", "codeword3", "unique_identity_token"]).any()
print(f"All semantic IDs are unique: {is_unique}")

semantic_id_to_item_id_df.to_csv("dataset/semantic_id_to_item_id.csv", index=False)