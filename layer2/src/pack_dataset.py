from pathlib import Path
import torch
from tokenizers import Tokenizer


TEXT_PATH = Path("../data/clean/clean.txt")
TOKENIZER_PATH = "../tokenizers/tokenizer.json"
OUTPUT_PATH = Path("../data/packed/train.pt")


SEQ_LEN = 64


def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    text = TEXT_PATH.read_text(encoding="utf-8")
    tokenizer = Tokenizer.from_file(TOKENIZER_PATH)

    bos_id = tokenizer.token_to_id('[BOS]') 
    eos_id = tokenizer.token_to_id('[EOS]')
    encoded = tokenizer.encode(text).ids
    tokens = [bos_id] + encoded + [eos_id]

    chunks = []
    
    for i in range(0 , len(tokens) - SEQ_LEN , SEQ_LEN):
        
        chunk = tokens[i:i+SEQ_LEN + 1]
        
        if len(chunk)  == SEQ_LEN + 1 :
            chunks.append(chunk)
    
    data =torch.tensor(chunks , dtype= torch.long)
    torch.save(data, OUTPUT_PATH)
    print(f'Saved packed data {OUTPUT_PATH}')
    print("Shape:", data.shape)
    print("Number of tokens:", len(tokens))
    print("Number of chunks:", len(chunks))
    
if __name__ == '__main__':
    main()