import math
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import List

class TransformerAudioClassifier(nn.Module):
    def __init__(self, input_size, num_classes, nhead=8, num_layers=6, dim_feedforward=2048):
        super().__init__()
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=nhead, dim_feedforward=dim_feedforward)
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = self.transformer(x)
        x = torch.mean(x, dim=1)
        x = self.fc(x)
        return x
    
def clones(module, N: int):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class Transformer(nn.Module):
    """Transformer architecture
    Args:
        num_of_class (dict): Number of class
        num_enc_block (int): Number of encoder block
        num_dec_block (int): Number of decoder block
        num_head (int): Number of self-attention head
        hidden (int): Hidden dimension of model
        fc_hidden (int): Hidden dimension of position-wise feedforward layer
        dropout (float): Dropout ratio
    """

    def __init__(
        self,
        num_of_class,
        num_enc_block,
        num_head,
        hidden,
        fc_hidden,
        dropout,
    ):
        super(Transformer, self).__init__()
        # token embedding + positional encoding
        self.src_embed = nn.Sequential(
            nn.Embedding(num_of_class, hidden), PositionalEncoding(hidden, dropout)
        )
        self.tgt_embed = nn.Sequential(
            nn.Embedding(num_of_class, hidden), PositionalEncoding(hidden, dropout)
        )

        # encoder and decoder
        self.encoder = Encoder(
            layer=EncoderLayer(
                num_head=num_head, hidden=hidden, fc_hidden=fc_hidden, dropout=dropout
            ),
            num_block=num_enc_block,
            hidden=hidden,
            dropout=dropout,
        )

        # generator
        self.generator = nn.Linear(hidden, num_of_class)

    def forward(self, src, src_mask):
        # encode
        src_embedding = self.src_embed(src)
        memory = self.encoder(src_embedding, src_mask)
        return self.generator(memory)


class Encoder(nn.Module):
    """Transformer encoder blocks
    Args:
        layer (nn.Module): Single encoder block
        num_block (int): Number of encoder block
        hidden (int): Hidden dimension of model
        dropout (float): Dropout ratio
    """

    def __init__(self, layer, num_block, hidden, dropout):
        super(Encoder, self).__init__()
        self.layers = clones(layer, num_block)
        self.layernorm = nn.LayerNorm(hidden)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.layernorm(x)


class EncoderLayer(nn.Module):
    """A single encoder block
    Args:
        num_head (int): Number of self-attention head
        hidden (int): Hidden dimension of model
        fc_hidden (int): Hidden dimension of position-wise feedforward layer
        dropout (float): Dropout ratio
    """

    def __init__(self, num_head, hidden, fc_hidden, dropout):
        super(EncoderLayer, self).__init__()
        self.multihead_attn = MultiHeadAttention(num_head, hidden, dropout)
        self.residual_1 = ResidualConnection(hidden, dropout)

        self.feedforward = PositionWiseFeedForward(hidden, fc_hidden, dropout)
        self.residual_2 = ResidualConnection(hidden, dropout)

    def forward(self, x, mask):
        # multi-head self-attention + residual connection
        # self.residual_k requires 'function' on second parameter, so use lambda
        x = self.residual_1(x, lambda x: self.multihead_attn(x, x, x, mask))

        # position-wise feedforward + residual connection
        x = self.residual_2(x, self.feedforward)
        return x


def attention(query, key, value, mask=None, dropout_fn=None):
    """Scaled dot product attention
    Args:
        query (Tensor): Query matrix -> [batch_size, num_head, hidden, d_k]
        key (Tensor): Key matrix -> [batch_size, num_head, hidden, d_k]
        value (Tensor): Value matrix -> [batch_size, num_head, hidden, d_k]
        mask (Tensor): Mask tensor which blocks calculating loss
        dropout_fn (nn.Module): Dropout function
    Returns:
        Calculated value matrix and attention distribution
    """
    d_k = query.shape[-1]
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores.masked_fill_(mask, -np.inf)
    p_attn = F.softmax(scores, dim=-1)

    if dropout_fn is not None:
        p_attn = dropout_fn(p_attn)
    context = torch.matmul(p_attn, value)
    return context, p_attn


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention module
    Args:
        num_head (int): Number of self-attention head
        hidden (int): Hidden dimension of model
        dropout (float): Dropout ratio
    """

    def __init__(self, num_head, hidden, dropout):
        super(MultiHeadAttention, self).__init__()
        assert hidden % num_head == 0
        self.d_k = hidden // num_head
        self.num_head = num_head
        self.dropout = nn.Dropout(dropout)

        self.W_q = nn.Linear(hidden, self.d_k * num_head)
        self.W_k = nn.Linear(hidden, self.d_k * num_head)
        self.W_v = nn.Linear(hidden, self.d_k * num_head)
        self.attn = None
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, hidden)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        b_size = query.shape[0]

        # make q, k, v matrice with multi-head
        # q,k,v: [batch_size, num_head, hidden, d_k]
        q = self.W_q(query).view(b_size, -1, self.num_head, self.d_k).transpose(1, 2)
        k = self.W_k(key).view(b_size, -1, self.num_head, self.d_k).transpose(1, 2)
        v = self.W_v(value).view(b_size, -1, self.num_head, self.d_k).transpose(1, 2)

        # calculate scaled dot product attention
        x, self.attn = attention(q, k, v, mask=mask, dropout_fn=self.dropout)

        # concatenate
        x = x.transpose(1, 2).contiguous().view(b_size, -1, self.num_head * self.d_k)
        return self.fc(x)


class ResidualConnection(nn.Module):
    """Residual connection with layer normalization
    Args:
        f_size (int): Feature size to normalize
        dropout (float): Dropout ratio
    """

    def __init__(self, f_size, dropout):
        super(ResidualConnection, self).__init__()
        self.layernorm = nn.LayerNorm(f_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.layernorm(x)))


class PositionWiseFeedForward(nn.Module):
    """Position-wise feedforward layer
    Args:
        hidden (int): Hidden dimension of model
        fc_hidden (int): Hidden dimension of position-wise feedforward layer
        dropout (float): Dropout ratio
    """

    def __init__(self, hidden, fc_hidden, dropout):
        super(PositionWiseFeedForward, self).__init__()
        self.fc_1 = nn.Linear(hidden, fc_hidden)
        self.fc_2 = nn.Linear(fc_hidden, hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc_2(self.dropout(F.relu(self.fc_1(x))))


class PositionalEncoding(nn.Module):
    """
    Args:
        hidden (int): Hidden dimension of model
        dropout (float): Dropout ratio
    """

    def __init__(self, hidden, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, hidden)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden, 2) * -(math.log(10000.0) / hidden))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.shape[1]].detach()
        return self.dropout(x)