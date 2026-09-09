import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2Tokenizer

VOCAB_SIZE = 50257
EMBED_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 2
FF_DIM = 256
MAX_LENGTH = 64

MODEL_PATH = "sentence_embedding_64d.pt"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


class PositionalEncoding(nn.Module):

    def __init__(self, dim, max_length):

        super().__init__()

        position = torch.arange(max_length).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, dim, 2)
            * (-torch.log(torch.tensor(10000.0)) / dim)
        )

        pe = torch.zeros(max_length, dim)

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


def mean_pool(hidden_states, attention_mask):

    mask = attention_mask.unsqueeze(-1).float()

    hidden_states = hidden_states * mask

    summed = hidden_states.sum(dim=1)

    count = mask.sum(dim=1).clamp(
        min=1e-9
    )

    return summed / count


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

        x = self.embedding(input_ids)

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


model = SentenceEmbeddingModel().to(DEVICE)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )
)

model.eval()

print("Model loaded successfully.")
print()


def get_embedding(sentence):

    tokens = tokenizer(
        sentence,
        padding=True,
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    input_ids = tokens["input_ids"].to(DEVICE)
    attention_mask = tokens["attention_mask"].to(DEVICE)

    with torch.no_grad():

        embedding = model(
            input_ids,
            attention_mask
        )

    return embedding


def similarity(sentence1, sentence2):

    embedding1 = get_embedding(sentence1)
    embedding2 = get_embedding(sentence2)

    cosine = F.cosine_similarity(
        embedding1,
        embedding2
    ).item()

    score = (cosine + 1.0) / 2.0

    sts_score = score * 5.0

    return cosine, score, sts_score


print("Semantic Similarity Test")
print("Type 'q' to quit.")
print()

while True:

    sentence1 = input("Sentence 1: ")

    if sentence1.lower() == "q":
        break

    sentence2 = input("Sentence 2: ")

    if sentence2.lower() == "q":
        break

    cosine, normalized, sts_score = similarity(
        sentence1,
        sentence2
    )

    print()
    print(f"Cosine similarity : {cosine:.4f}")
    print(f"Similarity (0-1)  : {normalized:.4f}")
    print(f"STS-B score (0-5) : {sts_score:.4f}")
    print()
