import torch.nn as nn

from .transformer import TransformerBlock
from .embedding import BERTEmbedding


class BERT(nn.Module):
    """
    BERT model : Bidirectional Encoder Representations from Transformers.
    """

    def __init__(self, vocab_size, hidden=768, n_layers=12, attn_heads=12, dropout=0.1):
        """ Bert 模型
        Args:
            vocab_size: 词表大小
            hidden: BERT 的 hidden size
            n_layers: Transformer 的层数
            attn_heads: Multi-head Attention 中的 head 数
            dropout: dropout rate
        """

        super().__init__()
        self.hidden = hidden
        self.n_layers = n_layers
        self.attn_heads = attn_heads

        # paper noted they used 4*hidden_size for ff_network_hidden_size
        self.feed_forward_hidden = hidden * 4

        # BERT的输入 embedding, 由 positional, segment, token embeddings 三部分组成
        self.embedding = BERTEmbedding(vocab_size=vocab_size, embed_size=hidden)

        # 多层的 Transformer (Encoder), 由多个 TransformerBlock 组成
        self.transformer_blocks = nn.ModuleList(
            [TransformerBlock(hidden, attn_heads, hidden * 4, dropout) for _ in range(n_layers)])

    def forward(self, x, segment_info):
        """
        x: [batch_size, seq_len]
        segment_info: [batch_size, seq_len]
        """

        # attention masking for padded token， 
        mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
        # [batch_size, 1, seq_len, seq_len]

        # embedding the indexed sequence to sequence of vectors
        x = self.embedding(x, segment_info)

        # running over multiple transformer blocks
        for transformer in self.transformer_blocks:
            x = transformer.forward(x, mask)

        return x
