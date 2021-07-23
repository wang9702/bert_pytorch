import torch.nn as nn


class SegmentEmbedding(nn.Embedding):
    def __init__(self, embed_size=512):
        """ 3 为 0:padding_idx, 1:sent_A, 2:sent_B """
        super().__init__(3, embed_size, padding_idx=0)
