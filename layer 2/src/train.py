from pathlib import Path
import torch
from torch.utils.data import DataLoader, TensorDataset
from tokenizers import Tokenizer
from tqdm import tqdm

from model import TinyGPT


TOKENIZER_PATH = "../tokenizers/tokenizer.json"
DATA_PATH = "../data/packed/train.pt"
CHECKPOINT_PATH = Path("../checkpoints/tiny_gpt.pt")

SEQ_LEN = 64
BATCH_SIZE = 8
EPOCHS = 20
LR = 3e-4

def main():
    CHECKPOINT_PATH.parent.mkdir(exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    vocab_size = tokenizer.get_vocab_size()

    data = torch.load(DATA_PATH)

    x = data[:, :-1]
    y = data[:, 1:]

    dataset = TensorDataset(x, y)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = TinyGPT(
        vocab_size=vocab_size,
        seq_len=SEQ_LEN,
        d_model=128,
        n_heads=4,
        n_layers=2,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for batch_x, batch_y in progress:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits, loss = model(batch_x, batch_y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            progress.set_postfix(loss=loss.item())

        avg_loss = total_loss / len(loader)
        print(f"Epoch {epoch + 1}: avg loss = {avg_loss:.4f}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_size": vocab_size,
            "seq_len": SEQ_LEN,
        },
        CHECKPOINT_PATH,
    )

    print(f"Saved checkpoint to {CHECKPOINT_PATH}")

if __name__ == "__main__":
    main()