import torch
import torch.nn as nn
import torch.nn.functional as F




###########################################
####              Encoder               ###
###########################################

class EncoderGRU(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self, device):
        return torch.zeros(1, 1, self.hidden_size, device=device)


class EncoderLSTM(nn.Module):
    def __init__(self, input_size, emb_size, hid_size, pad_token=-1):
        super().__init__()
        self.embedding_size = emb_size
        self.hidden_size = hid_size

        self.embedding = nn.Embedding(input_size, emb_size, padding_idx=pad_token)
        self.lstm = nn.LSTM(input_size=emb_size,
                            hidden_size=hid_size,
                            bidirectional=True)
        self.linear_h = nn.Linear(hid_size * 2, hid_size)
        self.linear_c = nn.Linear(hid_size * 2, hid_size)

    def forward(self, input_batch, input_lens):
        """
        :param input_batch: (s, b)
        :param input_lens: (b)

        :returns (s, b, 2h), ((1, b, h), (1, b, h))
        """
        batch_size = input_batch.shape[1]

        embedded = self.embedding(input_batch)  # (s, b) -> (s, b, h)
        output, (hidden_h, hidden_c) = self.lstm(embedded)

        hidden_h = hidden_h.transpose(1, 0)  # (2, b, h) -> (b, 2, h)
        hidden_h = hidden_h.reshape(batch_size, -1)  # (b, 2, h) -> (b, 2h)
        hidden_h = F.dropout(hidden_h, p=0.5, training=self.training)
        hidden_h = self.linear_h(hidden_h)  # (b, 2h) -> (b, h)
        hidden_h = F.relu(hidden_h)
        hidden_h = hidden_h.unsqueeze(0)  # (b, h) -> (1, b, h)

        hidden_c = hidden_c.transpose(1, 0)
        hidden_c = hidden_c.reshape(batch_size, -1)  # (b, 2, h) -> (b, 2h)
        hidden_c = F.dropout(hidden_c, p=0.5, training=self.training)
        hidden_c = self.linear_c(hidden_c)
        hidden_c = F.relu(hidden_c)
        hidden_c = hidden_c.unsqueeze(0)  # (b, h) -> (1, b, h)

        return output, (hidden_h, hidden_c)  # (s, b, 2h), ((1, b, h), (1, b, h))

