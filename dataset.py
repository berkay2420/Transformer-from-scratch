import torch
import torch.nn as nn
from torch.utils.data import Dataset

class BilingualDataset(Dataset):
    
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang, seq_len) ->None:
        super().__init__()

        self.ds = ds
        self.tokenizer_src  = tokenizer_src
        self.tokenizer_tgt = tokenizer_tgt
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang
        self.seq_len = seq_len
        
        self.sos_token = torch.Tensor([tokenizer_src.token_to_id(['[SOS]'])], dtype=torch.int64)
        self.eos_token = torch.Tensor([tokenizer_src.token_to_id(['[EOS]'])], dtype=torch.int64)
        self.pad_token = torch.Tensor([tokenizer_src.token_to_id(['[PAD]'])], dtype=torch.int64)

    def __len__(self):
      return int(len(self.ds))
    
    def __getitem__(self, index):
        src_target_pair = self.ds[index]
        src_text = src_target_pair['translation'][self.src_lang]
        tgt_text = src_target_pair['trainslation'][self.tgt_lang]

        enc_input_tokens = self.tokenizer_src.encode(src_text).ids
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        #adding padding tokens
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2 #Minus 2 because of the SOS & EOS tokens
        
        #we need model to learn to predict the End of sentence token 
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1 	

        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError('Sentence is too long')

        # Add SOS and EOS to the source text
        encoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(enc_input_tokens,dtype=torch.int64),
                self.eos_token,
                torch.tensor([self.pad_token] * enc_num_padding_tokens, dtype=torch.int64)
            ]
        )
        # Add SOS token tot the source text
        decoder_input = torch.cat(
            [
                self.sos_token,
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )
        # Add EOS to the label (what we expect as output from decoder)
        label = torch.cat(
            [
                torch.tensor(dec_input_tokens, dtype=torch.int64),
                self.sos_token,
                torch.tensor([self.pad_token] * dec_num_padding_tokens, dtype=torch.int64)
            ]
        )

        # everything must be in same size
        assert encoder_input.size(0) == self.seq_len
        assert decoder_input.size(0) == self.seq_len
        assert label.size(0) == self.seq_len

        return {
            "encoder_input": encoder_input, #seq_len
            "decoder_input": decoder_input, #seq_len

            # we dont want to use padding tokens in attention so we mask them
            "encoder_mask": (encoder_input != self.pad_token).unsqueeze(0).int(), #(1,1,seq_len)
            # But in decoder we also don't want words to see the next word 
            "decoder_mask": (decoder_input != self.pad_token).unsqueeze(0).int() & causal_mask(decoder_input.size(0)),
            # decoder_maksk = (1, seq_len) & (1, seq_len, seq_len)

            "label": label, #(seq_len)
            "src_text": src_text,
            "tgt_text": tgt_text
        }
    
def causal_mask(size):
    # Every value above the diagonal will be 1 and we need to use 0s (below the diagonal)
    # The logic behind is this that the model is learning to predict the next word in a sentence .
    # Thus the model should not see the what comes after the word
    """
        tensor([[[0, 1, 1, 1],
         [0, 0, 1, 1],
         [0, 0, 0, 1],
         [0, 0, 0, 0]]], dtype=torch.int32)
        -This is the attention graph and the digonal is where the word is paired with itself.
        -The mask ensures that tokens can only attend to previous tokens (not future ones).
        -This prevents the model from "seeing" future tokens while training, enforcing causality.
    """
    mask  = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    return mask == 0
