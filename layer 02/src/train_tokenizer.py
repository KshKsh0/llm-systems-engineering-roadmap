from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace # how to split words , in our case is space 


DATA_PATH = '../data/clean/clean.txt'
TOKENIZER_PATH = "../tokenizers/tokenizer.json"

def main():
    Path('../tokenizers').mkdir(exist_ok = True)
    
    tokenizer = Tokenizer(
        BPE(
            unk_token = '[UNK]')
        )
    
    tokenizer.pre_tokenizer  = Whitespace()
    trainer = BpeTrainer(
        
        vocab_size =1000,
        special_tokens = ["[UNK]", "[PAD]", "[BOS]", "[EOS]"]
    )
    
    tokenizer.train([DATA_PATH], trainer)
    tokenizer.save(TOKENIZER_PATH)
    
    print(f"Tokenizer saved to {TOKENIZER_PATH}")
    print("Vocab size:", tokenizer.get_vocab_size())
    
    
if __name__ == '__main__':
    main()

    