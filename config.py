from datasets import load_dataset

dataset = load_dataset(
    "sentence-transformers/stsb",
)

dataset.load_from_disk("./dataset")