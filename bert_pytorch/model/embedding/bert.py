import torch.nn as nn
from .token import TokenEmbedding
from .position import PositionalEmbedding
from .segment import SegmentEmbedding


class BERTEmbedding(nn.Module):
    """
    BERT Embedding 由以下三部分组成：
        1. TokenEmbedding : token embedding matrix
        2. PositionalEmbedding : 位置信息编码
        2. SegmentEmbedding : 句子信息编码, (sent_A:1, sent_B:2)
    """

    def __init__(self, vocab_size, embed_size, dropout=0.1):
        """
        Args:
            vocab_size: 词表大小
            embed_size: token embedding 的维度
            dropout: dropout rate
        """
        super().__init__()
        self.token = TokenEmbedding(vocab_size=vocab_size, embed_size=embed_size)
        self.position = PositionalEmbedding(d_model=self.token.embedding_dim)
        self.segment = SegmentEmbedding(embed_size=self.token.embedding_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_size = embed_size

    def forward(self, sequence, segment_label):
        x = self.token(sequence) + self.position(sequence) + self.segment(segment_label)
        return self.dropout(x)
