# src/evaluate.py

import torch
from torch.utils.data import DataLoader, TensorDataset
from tokenizers import Tokenizer

from model import TinyGPT

TOKENIZER_PATH = "../tokenizers/tokenizer.json"
DATA_PATH = "../data/packed/train.pt"
CHECKPOINT_PATH = "../checkpoints/tiny_gpt.pt"

BATCH_SIZE = 8

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    data = torch.load(DATA_PATH)

    x = data[:, :-1]
    y = data[:, 1:]

    loader = DataLoader(
        TensorDataset(x, y),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = TinyGPT(
        vocab_size=checkpoint["vocab_size"],
        seq_len=checkpoint["seq_len"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    total_loss = 0

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            _, loss = model(batch_x, batch_y)
            total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    print(f"Loss: {avg_loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}")

if __name__ == "__main__":
    main()