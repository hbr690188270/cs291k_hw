import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from typing import Optional, Any
import math

class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx if padding_idx is not None else 0
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size, embedding_dim, padding_idx
        )
        self.onnx_trace = False
        self.register_buffer("_float_tensor", torch.FloatTensor(1))
        self.max_positions = int(1e5)

    @staticmethod
    def get_embedding(
        num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None
    ):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(
            1
        ) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(
            num_embeddings, -1
        )
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb
    def make_positions(self, tensor, padding_idx: int,):
        """Replace non-padding symbols with their position numbers.

        Position numbers begin at padding_idx+1. Padding symbols are ignored.
        """
        # The series of casts and type-conversions here are carefully
        # balanced to both work with ONNX export and XLA. In particular XLA
        # prefers ints, cumsum defaults to output longs, and ONNX doesn't know
        # how to handle the dtype kwarg in cumsum.
        mask = tensor.ne(padding_idx).int()
        return (torch.cumsum(mask, dim=1).type_as(mask) * mask).long() + padding_idx

    def forward(
        self,
        input,
        timestep: Optional[Tensor] = None,
        positions: Optional[Any] = None,
    ):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input.size()
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos, self.embedding_dim, self.padding_idx
            )
        self.weights = self.weights.to(self._float_tensor)

        positions = self.make_positions(
            input, self.padding_idx,
        )
        return (
            self.weights.index_select(0, positions.view(-1))
            .view(bsz, seq_len, -1)
            .detach()
        )

class TransformerEmbedding(nn.Module):
    def __init__(self, vocab_size, hidden_size, pad_token_id, max_position_embeddings = 100, hidden_dropout_prob = 0.1, layer_norm_eps = 1e-12,
                pe_type = 'sin', layernorm_embedding = True):
        '''
        code reference: mainly from BertEmbedding in Huggingface's Transformers libarary, also refers to Transformer implementation in Fairseq
        '''
        ## pe_type:  sin/learn/none
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size, padding_idx= pad_token_id)
        self.pe_type = pe_type
        if pe_type == 'sin':
            self.position_embeddings = SinusoidalPositionalEmbedding(embedding_dim = hidden_size, padding_idx = pad_token_id)
        elif pe_type == 'learn':
            self.position_embeddings = nn.Embedding(max_position_embeddings, hidden_size)
        else:
            self.position_embeddings = None
        self.token_type_embeddings = nn.Embedding(2, hidden_size)
        if layernorm_embedding:
            self.LayerNorm = nn.LayerNorm(hidden_size, eps = layer_norm_eps)
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.register_buffer("position_ids", torch.arange(max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids, position_ids = None, token_type_ids = None, inputs_embeds = None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if self.pe_type == 'learn':
            if position_ids is None:
                position_ids = self.position_ids[:, 0 : seq_length]
            position_embeddings = self.position_embeddings(position_ids)
        elif self.pe_type == 'sin':
            position_embeddings = self.position_embeddings(input_ids)
        else:
            position_embeddings = None

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings

        if position_embeddings is not None:
            embeddings = embeddings + position_embeddings
        if self.LayerNorm is not None:
            embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings, inputs_embeds

