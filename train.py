from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from datasets import load_dataset
import torch.nn.functional as F
import torch.nn as nn
import torch

DATASET_NAME = "sentence-transformers/stsb"

MAX_LENGTH = 64

VOCAB_SIZE = 50257
EMBED_DIM = 64          # <-- Sentence embedding boyutu 64
NUM_HEADS = 4
NUM_LAYERS = 2
FF_DIM = 256

BATCH_SIZE = 32
EPOCHS = 10
LR = 3e-4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)

# Tokenizer (gpt 2)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

dataset = load_dataset(DATASET_NAME)

train_dataset = dataset["train"]
val_dataset = dataset["validation"]

print("Train:", len(train_dataset))
print("Validation:", len(val_dataset))

# Loading batch
def collate_fn(batch):
    sentences1 = [item["sentence1"] for item in batch]
    sentences2 = [item["sentence2"] for item in batch]
    scores = [float(item["score"]) for item in batch]

    tokens1 = tokenizer(
        sentences1,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    tokens2 = tokenizer(
        sentences2,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    return {
        "input_ids1": tokens1["input_ids"],
        "attention_mask1": tokens1["attention_mask"],

        "input_ids2": tokens2["input_ids"],
        "attention_mask2": tokens2["attention_mask"],

        "scores": torch.tensor(scores, dtype=torch.float32)
    }  

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_fn
)

# Positional encoding
class PositionalEncoding(nn.Module):

    def __init__(self, dim, max_length):

        super().__init__()

        position = torch.arange(
            max_length
        ).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, dim, 2)
            * (-torch.log(torch.tensor(10000.0)) / dim)
        )

        pe = torch.zeros(
            max_length,
            dim
        )

        pe[:, 0::2] = torch.sin(
            position * div_term
        )

        pe[:, 1::2] = torch.cos(
            position * div_term
        )

        pe = pe.unsqueeze(0)

        self.register_buffer(
            "pe",
            pe
        )

    def forward(self, x):

        return x + self.pe[:, :x.size(1)]


def mean_pool(
    hidden_states,
    attention_mask
):
    mask = attention_mask.unsqueeze(-1).float()

    hidden_states = hidden_states * mask

    summed = hidden_states.sum(dim=1)

    count = mask.sum(dim=1).clamp(
        min=1e-9
    )

    return summed / count

# Model
class SentenceEmbeddingModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.embedding = nn.Embedding(
            VOCAB_SIZE,
            EMBED_DIM
        )

        self.position = PositionalEncoding(
            EMBED_DIM,
            MAX_LENGTH
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,
            nhead=NUM_HEADS,
            dim_feedforward=FF_DIM,
            activation="gelu",
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=NUM_LAYERS
        )

        self.norm = nn.LayerNorm(
            EMBED_DIM
        )

    def forward(self, input_ids, attention_mask):
        x = self.embedding(
            input_ids
        )

        # [batch, seq, 64]
        x = self.position(x)

        x = self.encoder(
            x,
            src_key_padding_mask=(
                attention_mask == 0
            )
        )

        x = self.norm(x)

        x = mean_pool(
            x,
            attention_mask
        )

        x = F.normalize(
            x,
            p=2,
            dim=1
        )

        return x


model = SentenceEmbeddingModel().to(device=DEVICE)

print(model)

# optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LR
)

for epoch in range(EPOCHS):
    model.train()

    total_loss = 0

    for batch in train_loader:

        input_ids1 = batch["input_ids1"].to(DEVICE)
        mask1 = batch["attention_mask1"].to(DEVICE)

        input_ids2 = batch["input_ids2"].to(DEVICE)
        mask2 = batch["attention_mask2"].to(DEVICE)

        scores = batch["scores"].to(DEVICE)

        optimizer.zero_grad()

        # Sentence 1 -> 64 dim
        embedding1 = model(
            input_ids1,
            mask1
        )

        # Sentence 2 -> 64 dim
        embedding2 = model(
            input_ids2,
            mask2
        )

        # Cosine similarity
        similarity = F.cosine_similarity(
            embedding1,
            embedding2
        )

        similarity = (similarity + 1.0) / 2.0

        scores = scores / 5.0

        # Loss
        loss = F.mse_loss(
            similarity,
            scores
        )

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    average_loss = (
        total_loss / len(train_loader)
    )

    print(
        f"Epoch {epoch + 1}/{EPOCHS} "
        f"- Loss: {average_loss:.4f}"
    )

torch.save(
    model.state_dict(),
    "veytra-embed-base.pt"
)

print("Model saved successfully")