import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    '''
        An embedding layer that maps the token id into its corresponding word
        embeddings. The word embeddings are kept as fixed once initialized.
    '''
    def __init__(self, vocab, finetune_ebd=False, num_filters=50,
                 dropout=0.1):
        super(TextCNN, self).__init__()

        # get word embedding layer
        self.vocab_size, self.embedding_dim = vocab.vectors.size()
        self.embedding_layer = nn.Embedding(
                self.vocab_size, self.embedding_dim)
        self.embedding_layer.weight.data = vocab.vectors

        self.finetune_ebd = finetune_ebd

        if self.finetune_ebd:
            self.embedding_layer.weight.requires_grad = True
        else:
            self.embedding_layer.weight.requires_grad = False

        # get cnn layer
        self.convs = nn.ModuleList([nn.Conv1d(
            in_channels=self.embedding_dim, out_channels=num_filters,
            kernel_size=k) for k in [3, 4, 5]])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        '''
            @param text: batch_size * max_text_len
            @return output: batch_size * embedding_dim
        '''
        x = self.embedding_layer(x)
        x = self.dropout(x)

        x = x.permute(0, 2, 1)  # batch_size, embedding_dim, doc_len
        x = x.contiguous()

        x = [conv(x) for conv in self.convs]

        # max pool over time. Resulting dimension is
        # [batch_size, num_filters] * len(filter_size)
        x = [F.max_pool1d(sub_x, sub_x.size(2)).squeeze(2) for sub_x in x]

        # concatenate along all filters. Resulting dimension is
        # [batch_size, num_filters_total]
        x = torch.cat(x, 1)
        x = F.relu(x)

        return x
