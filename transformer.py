#########################################################################################
#											#
#   CODE BY: JESSY LIAO / JOSEPH KIM / COURTNEY RICHARDSON / MATT CLOUGH		#
#   BASE CODE FOUND HERE: https://github.com/SamLynnEvans/Transformer			#
#   CSCI470: FINAL PROJECT							        #
#											#
#########################################################################################
# For Transformer, Encoder, Decoder
import sys
import copy
import torch
import torch.nn as nn

# For Embedder, PositionalEncoder
import math
from torch.autograd import Variable

# For MultiHeadedAttention
import torch.nn.functional as F

###########################################################################################
#                                                                                       
# MODEL
#                                                                                       
###########################################################################################
def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

# MAIN: Transformer class, calls all other class objects
class Transformer(nn.Module):
    """
    paramters:
        - source_vocab_size: LENGTH of torch vocab object 
        - target_vocab_size: LENGTH of torch vocab object
        - d_model: dimension used to calculate attention
        - N: number of encoding/decoding layers
        - heads: number of partition in the word vector
        - dropout: ???
    """
    def __init__(self, source_vocab_size, target_vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.encoder = Encoder(source_vocab_size, d_model, N, heads, dropout)
        self.decoder = Decoder(target_vocab_size, d_model, N, heads, dropout) 
        self.out = nn.Linear(d_model, target_vocab_size)

    """
    parameters:
        - source: ???
        - target: ???
        - source_mask: ???
        - target_mask: ???
    """
    def forward(self, source, target, source_mask, target_mask):
        if source is None:
            print("transformer")
            quit()
        e_outputs = self.encoder(source, source_mask)
        d_output = self.decoder(target, e_outputs, source_mask, target_mask)
        
        output = self.out(d_output)
        return output

class Encoder(nn.Module):
    # Paramters same as transformer model, except it uses source vocab size
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N 
        self.embed = Embedder(vocab_size, d_model) # Embedding layer: (meaning of word)
        self.pe = PositionalEncoder(d_model, dropout=dropout) # PE layer: (position in sentence)
        self.layers = get_clones(EncoderLayer(d_model, heads, dropout), N) # Get N copy of layer
        self.norm = Norm(d_model) # Normalize: calibrate data for every iteration of layer
    def forward(self, source, mask):
        if source is None:
            print("encoder")
            quit()
        x = self.embed(source)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, mask)
        return self.norm(x)

class Decoder(nn.Module):
    # Parameters same as transformer model, except it uses target vocab size
    def __init__(self, vocab_size, d_model, N, heads, dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size, d_model) 
        self.pe = PositionalEncoder(d_model, dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model, heads, dropout), N)
        self.norm = Norm(d_model)
    def forward(self, target, e_outputs, source_mask, target_mask):
        x = self.embed(target)
        x = self.pe(x)
        for i in range(self.N):
            x = self.layers[i](x, e_outputs, source_mask, target_mask)
        return self.norm(x)


###########################################################################################
#                                                                                       
# EMBED
#                                                                                       
###########################################################################################
class Embedder(nn.Module):
    """
    parameters:
        - vocab_size: can be target or source
        - d_model: dimension of the word embedding vector
    """
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model) # Torch Class
    def forward(self, x):
        if x is None:
            print("embedder")
            quit()
        return self.embed(x)

class PositionalEncoder(nn.Module):
    """
    parameters:
        - d_model: dimensions of the word embedding vector
        - max_seq_len: number of words in a sentence
        - dropout: prevents overfitting by removing some data
    """
    def __init__(self, d_model, max_seq_len=200, dropout=0.1): 
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        """
        Creates a constant 2d matrix
            - pos: order in the sentence
            - i: position in embedding vector
        """
        pe = torch.zeros(max_seq_len, d_model)

        for pos in range(max_seq_len):
            for i in range(0, d_model, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i)/d_model)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1))/d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe',pe)

    def forward(self, x):
        if x is None:
            print("pe")
            quit()
        # make the embedding relatively larger (this is so meaning of the word has more impact than the position of the word in the sentence. 
        x = x * math.sqrt(self.d_model)

        # add constant to embedding ???
        seq_len = x.size(1)
        pe = Variable(self.pe[:,:seq_len], requires_grad=False)
        if x.is_cuda:
            pe.cuda()
        x = x + pe
        return self.dropout(x)


###########################################################################################
#                                                                                       
# LAYERS
#                                                                                       
###########################################################################################
class EncoderLayer(nn.Module):
    """
    parameters:
        d_model: dimension of embedding vector
        heads: number of split in embedding vector
        dropout: ???
    """
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        # I think this can be done in one norm and one dropout
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        
        # Pass the variables into attention function
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        # Feed forward
        self.ff = FeedForward(d_model, dropout=dropout)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        if x is None:
            print("encoderlayer")
            quit()
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn(x2,x2,x2,mask))
        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.ff(x2))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        """
        parameters: same as decoder layer
        """
        super().__init__()
        # I think this can be done in one norm/dropout as well
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

        # Attention 1 is for the output from the encoder which is 
        self.attn_1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        # Attention 2 is the output embedding (this starts at random value, and tunes it so we know what it means)
        self.attn_2 = MultiHeadAttention(heads, d_model, dropout=dropout)

        self.ff = FeedForward(d_model, dropout=dropout)

    def forward(self, x, e_output, source_mask, target_mask): 
        if x is None:
            print("decoderlayer")
            quit()
        x2 = self.norm_1(x)
        x = x + self.dropout_1(self.attn_1(x2, x2, x2, target_mask))

        x2 = self.norm_2(x)
        x = x + self.dropout_2(self.attn_2(x2, e_output, e_output, source_mask))

        x2 = self.norm_3(x)
        x = x + self.dropout_3(self.ff(x2))

        return x # this is an embedding vector (now we know what the word means)


###########################################################################################
#                                                                                       
# SUB LAYERS
#                                                                                       
###########################################################################################
# This is just the math to calculate attention (This is a helper method for MultiHeadedAttention)
def attention(q, k, v, d_k, mask=None, dropout=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None: 
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)

    scores = F.softmax(scores, dim=-1)

    if dropout is not None: 
        scores = dropout(scores)

    output = torch.matmul(scores, v)
    return output 

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model, dropout = 0.1):
        super().__init__()

        # variables used for the attention function
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads

        # each of these represent a word
        # linearilizes it 
        self.q_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        bs = q.size(0)

        # perform linear operations and split into N heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # tranpose to get dimensions bs * N * sl * d_model
        k = k.transpose(1,2)
        q = q.transpose(1,2)
        v = v.transpose(1,2)

        # calculate attention using function we defined earlier
        scores = attention(q, k, v, self.d_k, mask, self.dropout)

        #concatenate heads and put through final linear layer
        concat = scores.transpose(1,2).contiguous().view(bs, -1, self.d_model)

        output = self.out(concat)

        return output

class Norm(nn.Module): 
    def __init__(self, d_model, eps = 1e-6):
        super().__init__()

        self.size = d_model

        # create two learnable parameters to calibrate normalization 
        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))

        self.eps = eps

    def forward(self, x):
        if x is None:
            print("norm")
            quit()
        norm = self.alpha * (x - x.mean(dim=-1, keepdim=True)) / (x.std(dim=-1, keepdim=True) + self.eps) + self.bias
        return norm

class FeedForward(nn.Module): 
    def __init__(self, d_model, d_ff=2048, dropout=0.1): 
        super().__init__()

        # Set d_ff as a default to 2048
        # This takes the word vector size and the default d_ff and linearize the value
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        if x is None:
            print("ff")
            quit()
        # Performs relu calculation and linearizes that value and returns it
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)
        return x
