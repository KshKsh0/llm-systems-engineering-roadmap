import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyGPT(nn.Module):
    
    def __init__(self, vocab_size:int,
            seq_len:int,
            
            d_model: int = 128, #This is the embedding size, also called the hidden dimension.
            
            n_heads: int = 4,
            
            n_layers: int = 2,
            
            dropout: float = 0.1,
            ):
        
        super().__init__()
        
        self.seq_len = seq_len
        self.token_embedding  = nn.Embedding(vocab_size ,d_model)
        self.position_embedding =nn.Embedding(seq_len, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout, batch_first=True, 
            activation="gelu", ) # GELU is a common activation function used in GPT-style models
        self.blocks = nn.TransformerEncoder( encoder_layer, num_layers=n_layers, )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        

    def forward(self ,input_ids , targets=None):
        batch_size, seq_len = input_ids.shape
        positions=torch.arange(seq_len ,device =input_ids.device )
        token_emb = self.token_embedding(input_ids) 
        pos_emb = self.position_embedding(positions)
        
        x = token_emb + pos_emb
        
        causal_mask = torch.triu( torch.ones(seq_len, seq_len, device=input_ids.device), diagonal=1 ).bool()  #  to make sure the model cant see next token only previose ones , CASUAL MASKING 

        x = self.blocks(x, mask=causal_mask) 
        x = self.ln_f(x)
        
        logits = self.lm_head(x) 
        loss = None 
        if targets is not None: 
            loss = F.cross_entropy( logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ) 
        return logits, loss
                
        
        
        