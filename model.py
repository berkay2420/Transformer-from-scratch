import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size:int):
        super().__init__()
        self.d_model = d_model #size of emnedding vector
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model) # convert categorical data, such as words or tokens, into dense vector representations (embeddings). 

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model) #this calculation was also used in the paper

class PositionalEncoding(nn.Module):
  
    def __init__(self, d_model: int, seq_len, dropout: float ) -> None:
        super().__init__()
        self.d_model = d_model  
        self.seq_len = seq_len
        self.droput = nn.Dropout(dropout)

        # Initialize a matrix to store positional encodings (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)

        # Create a position vector (seq_len, 1) for each position in the sequence
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model,2).float() * (-math.log(10000.0) / d_model))

        # Apply sine to even indices and cosine to odd indices in the embedding dimension
        pe[:, 0::2] = torch.sin(position * div_term)  # Even positions
        pe[:, 1::2] = torch.cos(position * div_term)  # Odd positions

        pe = pe.unsqueeze(0) # (1, seq_len, d_model)

        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad__(False)
        return self.droput(x)
  

class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10** - 6 ) -> None:
        super().__init__()
        self.eps = eps # for avoiding division by zero 
        self.alpha = nn.Parameter(torch.ones(1))  # Learnable scaling parameter (initialized to 1)
        self.bias = nn.Parameter(torch.zeros(1))  # Learnable bias parameter (initialized to 0)

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True)
        std = x.std(dim = -1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForwardBlock(nn.Module):

    def __init__(self, d_model:int, d_ff:int, droput: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff) 
        self.droput = nn.Dropout(droput)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
      # (Batch, seq_len, d_model) --> (Batch, seq_len, d_model) --> (Batch, seq_len, d_model)
      return self.linear_2(self.droput(torch.relu(self.linear_1(x))))
    
class MultiHeadAttentionBlock(nn.Module):

    def __init__(self, d_model:int, h:int, droput: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divideble by h"

        self.d_k = d_model // h
        self.w_q = nn.Linear(d_model, d_model) #Wq
        self.w_k = nn.Linear(d_model, d_model) #Wk
        self.w_v = nn.Linear(d_model, d_model) #Wv

        self.w_o = nn.Linear(d_model, d_model) #Wo
        self.dropout = nn.Dropout(droput)
    
    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        # (Batch, h, seq_len, d_k) --> (Batch, h, seq_len, seq_len)
        attention_scores = (query @ key.transpose(-2,-1)) / math.sqrt(d_k)
        
        if mask is not None:
          attention_scores.masked_fill_(mask ==0, -1e9) 
          # Before applying softmax, we assign a very large negative value (-∞ approx.) 
          # to masked positions, ensuring they get near-zero probability.
        attention_scores = attention_scores.softmax(dim=-1) # (Batch, h, seq_len, seq_len)
        if dropout is not None:
          attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores
                # The weighted value tensor after applying attention scores., The attention scores matrix.
    
    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (Batch, seq_len, d_model)-->(Batc, seq_len, d_model)
        key = self.w_k(k)
        value = self.w_v(v)

        #(Batch, SeqLen, d_model) --> (Batch, SeqLen, h, d_k) --> (Batch, h, SeqLen, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1,2) #(BatchSize,SeqLen,AttentionN,Dimensions)
        # the tensor is now ready for scaled dot-product attention across multiple heads

        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)
        value = value.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1,2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # (Batch, h, seq_len, d_k) ---> (Batch, seq_len, h, d_k) ---> (Batch, seq_len, d_model)
        x = x.transpose(1,2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

        return self.w_o(x)

class ResidualConnection(nn.Module):

    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))
    

class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block:MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) ->None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        # Source Mask is the mask for the input of the encoder, we want to hide the interaction of the padding word with other words

        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x
  
class Encoder(nn.Module):

    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    
    def forward(self, x, mask):
        for layer in self.layers:
          x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList(ResidualConnection(dropout) for _ in range(3))

    def forward(self, x, encoder_output, src_maks, tgt_mask):
        # source mask = mask for encoder
        # target mask = mask for decoder
        # source language(english) vs target language(italian)

        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_maks))
        x = self.residual_connections[2](x, self.feed_forward_block)
        return x

class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers:
          x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)

class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # (Batch, Seq_len, d_model) ---> (Batch, seq_len, vocab_Size)
        return torch.log_softmax(self.proj(x), dim = -1)
  
class Transformer(nn.Module):

    def __init__(self, encoder:Encoder, decoder:Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, 
                src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer ) -> None:
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos
        self.projection_layer = projection_layer
    
    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decode(self, encoder_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.encoder(tgt, encoder_output, src_mask, tgt_mask)
    
    def project(self, x):
        return self.projection_layer(x)
  
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, 
                      d_model: int = 512, N: int = 6, h: int = 8, dropout: float = 0.01, d_ff: int =2048) -> Transformer:
    # creating the embedding layers 
    src_embedding = InputEmbeddings(d_model, src_vocab_size)
    tgt_embedding = InputEmbeddings(d_model, tgt_vocab_size)

    #creating the positional encoding layers
    src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
    tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

    #creating the encoding blocks
    encoder_blocks = []
    for _ in range(N):
      encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
      feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
      encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
      encoder_blocks.append(encoder_block)
    
    #creating the decoder blocks
    decoder_blocks = []
    for _ in range(N):
      decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
      decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
      feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
      decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
      decoder_blocks.append(decoder_block)
    
    #create the encoder and decoder
    encoder = Encoder(nn.ModuleList(encoder_blocks))
    decoder = Decoder(nn.ModuleList(decoder_blocks))

    #projection layer
    projection_layer = ProjectionLayer(d_model, tgt_vocab_size)#converting d_model into vocab size

    #Build the transformer
    transformer = Transformer(encoder, decoder, src_embedding, tgt_embedding, src_pos, tgt_pos, projection_layer)

    # Initialize the parameters
    for p in transformer.parameters():
      if p.dim() > 1:
        nn.init.xavier_uniform_(p)
    
    return transformer