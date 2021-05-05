import os.path as osp

import torch
import torch.nn as nn
import torch.nn.functional as F




class DecoderBaseModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def load_weights(self, load_m_dir=None, load_m_file_name='decoder.pth',):
        if load_m_dir is not None:
            dec_path = osp.join(load_m_dir, load_m_file_name)
            param = torch.load(dec_path)
            self.load_state_dict(param)
            print(f'[info] {load_m_file_name} loaded !')

    def save(self, save_f_path='_logs/test/decoder.pth',):
        torch.save(self.state_dict(), save_f_path)




###########################################
##                 Decoder               ##
###########################################

class DecoderLSTM(DecoderBaseModel):
    def __init__(self, emb_size, hidden_size, output_size, pad_token=-1):
        super().__init__()

        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_token)
        self.lstm = nn.LSTMCell(emb_size, hidden_size)
        self.out_w = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        """
        :param input: (b)
        :param hidden: ((b,h), (b,h))
        :return: (b,o), (b,h)
        """
        embedded = self.embedding(input)  # (b) -> (b,e)
        decoder_output, hidden = self.lstm(embedded, hidden)  # (b,e),((b,h),(b,h)) -> (b,h),((b,h),(b,h))
        output = self.out_w(decoder_output)  # (b,h) -> (b,o)
        output = F.log_softmax(output, dim=1)

        return output, hidden  # (b,o), (b,h)


class DecoderGRU(DecoderBaseModel):
    def __init__(self, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, 1, -1)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.softmax(self.out(output[0]))
        return output, hidden

    def initHidden(self, device):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)





###########################################
##          Attention  Decoder           ##
###########################################

class AttnDecoderLSTM1(DecoderBaseModel):
    def __init__(self, emb_size, hidden_size, output_size, device, max_length=20):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_length = max_length

        self.embedding = nn.Embedding(output_size, emb_size)
        self.attn = nn.Linear(emb_size+2*hidden_size, max_length)
        self.attn_combine = nn.Linear(emb_size+2*hidden_size, hidden_size)

        self.lstm = nn.LSTMCell(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)

        self.device = device
    
    def forward(self, input, hidden, encoder_outputs):
        """
        :param input: (b)
        :param hidden: ((b, h), (b, h))
        :param encoder_outputs: (il, b, 2h)
        :return: (b,o), ((b,h),(b,h)), (b,il)
        """
        input_length = encoder_outputs.shape[0]
        #padding
        encoder_outputs = torch.cat([
            encoder_outputs,
            torch.zeros(
                self.max_length - input_length,
                encoder_outputs.shape[1],
                encoder_outputs.shape[2],
                device=self.device
            )
        ], dim=0)  # (il,b,2h), (ml-il,b,2h) -> (ml,b,2h)
        drop_encoder_outputs = F.dropout(encoder_outputs, p=0.1, training=self.training)
        
        # embedding
        embedded = self.embedding(input)  # (b) -> (b,e)
        embedded = F.dropout(embedded, p=0.5, training=self.training)
    
        emb_hidden = torch.cat([embedded, hidden[0], hidden[1]], dim=1)  # (b,e),((b,h),(b,h)) -> (b,e+2h)

        attn_weights = self.attn(emb_hidden)  # (b,e+2h) -> (b,ml)
        attn_weights = F.softmax(attn_weights, dim=1)

        attn_applied = torch.bmm(
            attn_weights.unsqueeze(1),  # (b, 1, ml)
            drop_encoder_outputs.transpose(0, 1)  # (b, ml, 2h)
        )  # -> (b, 1, 2h)

        attn_applied = F.dropout(attn_applied, p=0.1, training=self.training)
        output = torch.cat((embedded, attn_applied.squeeze(1)), 1)  # ((b,e),(b,2h)) -> (b,e+2h)
        output = self.attn_combine(output)  # (b,e+2h) -> (b,h)
        output = F.dropout(output, p=0.5, training=self.training)

        output = F.relu(output)
        hidden = self.lstm(output, hidden)  # (b,h),((b,h),(b,h)) -> (b,h)((b,h),(b,h))

        output = F.log_softmax(self.out(hidden[0]), dim=1)  # (b,h) -> (b,o)
        return output, hidden, attn_weights  # (b,o),(b,h),(b,il)


class AttnDecoderLSTM2(DecoderBaseModel):
    def __init__(self, emb_size, hidden_size, attn_size, output_size, device,
                        pad_token=-1, max_length=10):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.embedding = nn.Embedding(output_size, emb_size, padding_idx=pad_token)
        self.lstm = nn.LSTMCell(emb_size, hidden_size)

        self.score_w = nn.Linear(2*hidden_size, 2*hidden_size)
        self.attn_w = nn.Linear(4*hidden_size, attn_size)
        self.out_w = nn.Linear(attn_size, output_size)

        self.device = device

    def forward(self, input, hidden, encoder_outputs):
        """
        :param: input: (b)
        :param: hidden: ((b,h),(b,h))
        :param: encoder_outputs: (il,b,2h)

        :return: (b,o), ((b,h),(b,h)), (b,il)
        """
        
        embedded = self.embedding(input)  # (b) -> (b,e)
        embedded = F.dropout(embedded, p=0.5, training=self.training)
        
        hidden = self.lstm(embedded, hidden)  # (b,e),((b,h),(b,h)) -> ((b,h),(b,h))
        decoder_output = torch.cat(hidden, dim=1)  # ((b,h),(b,h)) -> (b,2h)
        decoder_output = F.dropout(decoder_output, p=0.5, training=self.training)

        # score
        score = self.score_w(decoder_output)  # (b,2h) -> (b,2h)
        scores = torch.bmm(
            encoder_outputs.transpose(0, 1),  # (b,il,2h)
            score.unsqueeze(2)  # (b,2h,1)
        )  # (b,il,1)
        attn_weights = F.softmax(scores, dim=1)  # (b,il,1)

        # context
        context = torch.bmm(
            attn_weights.transpose(1, 2),  # (b,1,il)
            encoder_outputs.transpose(0, 1)  # (b,il,2h)
        )  # (b,1,2h)
        context = context.squeeze(1)  # (b,1,2h) -> (b,2h)

        concat = torch.cat((context, decoder_output), dim=1)  # ((b,2h),(b,2h)) -> (b,4h)
        #concat = F.dropout(concat, p=0.5, training=self.training)

        attentional = self.attn_w(concat)  # (b,4h) -> (b,a)
        attentional = torch.tanh(attentional)
        #attentional = F.dropout(attentional, p=0.5, training=self.training)

        output = self.out_w(attentional)  # (b,a) -> (b,o)
        output = F.log_softmax(output, dim=1)

        return output, hidden, attn_weights.squeeze(2)  # (b,o), ((b,h),(b,h)), (b,il)




class AttnDecoderGRU(DecoderBaseModel):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=10):
        super(AttnDecoderGRU, self).__init__()

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)


    def forward(self, input, hidden, enc_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 enc_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self, device):
        return torch.zeros(1, 1, self.hidden_size, device=device)


