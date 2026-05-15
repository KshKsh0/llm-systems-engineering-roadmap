# src/generate.py

import torch
from tokenizers import Tokenizer

from model import TinyGPT

TOKENIZER_PATH = "../tokenizers/tokenizer.json"
CHECKPOINT_PATH = "../checkpoints/tiny_gpt.pt"

MAX_NEW_TOKENS = 50
TEMPERATURE = 0.8

def generate(model, input_ids, max_new_tokens, temperature):
    model.eval()

    for _ in range(max_new_tokens):
        input_context = input_ids[:, -model.seq_len:]

        with torch.no_grad():
            logits, _ = model(input_context)

        next_logits = logits[:, -1, :] / temperature
        probs = torch.softmax(next_logits, dim=-1)

        next_token = torch.multinomial(probs, num_samples=1)

        input_ids = torch.cat([input_ids, next_token], dim=1)

    return input_ids

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    model = TinyGPT(
        vocab_size=checkpoint["vocab_size"],
        seq_len=checkpoint["seq_len"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])

    prompt = "Large language models"

    encoded = tokenizer.encode(prompt).ids
    input_ids = torch.tensor([encoded], dtype=torch.long).to(device)

    output_ids = generate(
        model,
        input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
    )

    output_text = tokenizer.decode(output_ids[0].tolist())

    print('generated text: ' , output_text)

if __name__ == "__main__":
    main()




#garbage output lol