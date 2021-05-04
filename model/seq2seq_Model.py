from abc import ABCMeta, abstractmethod
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from abc_ml.abc_model import ModelABC
from abc_ml.abc_tokenizer import Seq2Seq_TokenizerABC

from model.rnn_model.encoderRNN import EncoderGRU, EncoderLSTM
from model.rnn_model.decoderRNN import DecoderGRU, DecoderLSTM, AttnDecoderGRU, AttnDecoderLSTM, AttnDecoderLSTM2


SOS_token = 0
EOS_token = 1


"""
Pytorch チュートリアル（Seq2Seq）
・ https://torch.classcat.com/2018/05/15/pytorch-tutorial-intermediate-seq2seq-translation/
"""





class Seq2SeqTranslate_ptTokenizer(Seq2Seq_TokenizerABC):
    def __init__(self, input_lang, output_lang, device, tensor_type="pt"):
        super().__init__(input_lang, output_lang, device, tensor_type)
    
    # 訓練データの準備
    # 訓練するために、各ペアについて入力 tensor (入力センテンスの単語のインデックス) とターゲット tensor (ターゲット・センテンスの単語のインデックス) が必要です。これらのベクトルを作成する一方で両者のシーケンスに EOS トークンを追加します。
    def sentence_to_indexs(self, sentence, is_input:bool):
        lang = self.input_lang if is_input else self.output_lang
        return [lang.word2index[word] for word in sentence.split(' ')]

    def get_tensor_from_sentence(self, sentence, is_input:bool):
        indexes = self.sentence_to_indexs(sentence, is_input)
        indexes.append(EOS_token)
        return torch.tensor(indexes, dtype=torch.long, device=self.device).view(-1, 1)

    def get_tensors_from_pair(self, pair):
        input_tensor = self.get_tensor_from_sentence(pair[0], is_input=True)
        target_tensor = self.get_tensor_from_sentence(pair[1], is_input=False)
        return (input_tensor, target_tensor)



class Seq2Seq_LSTM_Attn_ptModel():
    # def __init__(self, input_n, output_n, hidden_size, tokenizer, device, dropout_p=0.1, max_length=10):
    #     self.MAX_LENGTH = max_length
    #     self.device = device

    #     self.teacher_forcing_ratio = 0.5
    #     self.encoder = EncoderGRU(input_n, hidden_size).to(self.device)
    #     self.decoder = AttnDecoderGRU(hidden_size, output_n, dropout_p=dropout_p).to(device)

    #     self.tokenizer = tokenizer
    
    def _test(self):
        batch_size = 10
        emb_size = 8
        hid_size = 12
        attn_size = 9

        # test encoder
        test_encoder = EncoderLSTM(input_lang.n_words, emb_size, hid_size).to(device)
        for input_batch, input_lens, output_batch, output_lens in generate_batch(train_pairs, batch_size):
            break
        input_batch.shape, input_lens.shape

        encoder_outputs, (hidden_h, hidden_c) = test_encoder(input_batch, input_lens)
        encoder_outputs.shape, hidden_h.shape, hidden_c.shape

        hidden = (hidden_h.squeeze(0), hidden_c.squeeze(0))
        hidden[0].shape, hidden[1].shape

        test_decoder1 = AttnDecoderLSTM1(emb_size, hid_size, output_lang.n_words, max_length=MAX_LENGTH).to(device)
        decoder_input = torch.tensor([SOS_token] * batch_size, device=device)
        decoder_outputs, hidden, attn_weights = test_decoder1(decoder_input, hidden, encoder_outputs)
        decoder_outputs.shape, hidden[0].shape, hidden[1].shape, attn_weights.shape

        criterion = nn.NLLLoss(ignore_index=PAD_token)
        loss = criterion(decoder_outputs, output_batch[0])
        loss.item()


    def fit_batch(self, input_batch, input_lens, target_batch, target_lens,
                    encoder, decoder, optimizer, criterion,
                    teacher_forcing_ratio=0.5):
        loss = 0
        optimizer.zero_grad()

        batch_size = input_batch.shape[1]
        target_length = target_lens.max().item()

        encoder_outputs, encoder_hidden = encoder(input_batch, input_lens)  # (s, b, 2h), ((1, b, h), (1, b, h))
        
        decoder_input = torch.tensor([[SOS_token] * batch_size], device=device)  # (1, b)
        decoder_inputs = torch.cat([decoder_input, target_batch], dim=0)  # (1,b), (n,b) -> (n+1, b)
        decoder_hidden = (encoder_hidden[0].squeeze(0), encoder_hidden[1].squeeze(0))
        
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
        
            for di in range(target_length):
                decoder_output, decoder_hidden, attention = decoder(
                    decoder_inputs[di], decoder_hidden, encoder_outputs)

                loss += criterion(decoder_output, decoder_inputs[di+1])
        else:
            decoder_input = decoder_inputs[0]
            for di in range(target_length):
                decoder_output, decoder_hidden, attention = decoder(
                    decoder_input, decoder_hidden, encoder_outputs)

                loss += criterion(decoder_output, decoder_inputs[di+1])

                _, topi = decoder_output.topk(1)  # (b,odim) -> (b,1)
                decoder_input = topi.squeeze(1).detach() 

        loss.backward()

        optimizer.step()

        return loss.item() / target_length


    # def predict(self, sentence):
    #     with torch.no_grad():
    #         input_tensor = self.tokenizer.get_tensor_from_sentence(sentence, is_input=True)
    #         input_length = input_tensor.size()[0]
    #         enc_hidden = self.encoder.initHidden(self.device)

    #         enc_outputs = torch.zeros(self.MAX_LENGTH, self.encoder.hidden_size, device=self.device)

    #         for ei in range(input_length):
    #             enc_output, enc_hidden = self.encoder(input_tensor[ei],
    #                                                     enc_hidden)
    #             enc_outputs[ei] += enc_output[0, 0]

    #         decoder_input = torch.tensor([[SOS_token]], device=self.device)  # SOS

    #         decoder_hidden = enc_hidden

    #         decoded_words = []
    #         decoder_attentions = torch.zeros(self.MAX_LENGTH, self.MAX_LENGTH)

    #         for di in range(self.MAX_LENGTH):
    #             decoder_output, decoder_hidden, decoder_attention = self.decoder(
    #                 decoder_input, decoder_hidden, enc_outputs)
    #             decoder_attentions[di] = decoder_attention.data
    #             topv, topi = decoder_output.data.topk(1)
    #             if topi.item() == EOS_token:
    #                 decoded_words.append('')
    #                 break
    #             else:
    #                 decoded_words.append(self.tokenizer.output_lang.index2word[topi.item()])

    #             decoder_input = topi.squeeze().detach()

    #         return decoded_words, decoder_attentions[:di + 1]





class Seq2Seq_GRU_Attn_ptModel():
    def __init__(self, input_n, output_n, hidden_size, tokenizer, device, dropout_p=0.1, max_length=10):
        self.MAX_LENGTH = max_length
        self.device = device

        self.teacher_forcing_ratio = 0.5
        self.encoder = EncoderGRU(input_n, hidden_size).to(self.device)
        self.decoder = AttnDecoderGRU(hidden_size, output_n, dropout_p=dropout_p).to(device)

        self.tokenizer = tokenizer

    def fit(self, input_tensor, target_tensor,
                enc_optimizer, dec_optimizer, criterion):
        enc_hidden = self.encoder.initHidden(self.device)

        enc_optimizer.zero_grad()
        dec_optimizer.zero_grad()

        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)

        enc_outputs = torch.zeros(self.MAX_LENGTH, self.encoder.hidden_size, device=self.device)

        loss = 0

        for ei in range(input_length):
            enc_output, enc_hidden = self.encoder(
                input_tensor[ei], enc_hidden)
            enc_outputs[ei] = enc_output[0, 0]

        dec_input = torch.tensor([[SOS_token]], device=self.device)

        dec_hidden = enc_hidden

        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                dec_output, dec_hidden, dec_attention = self.decoder(
                    dec_input, dec_hidden, enc_outputs)
                loss += criterion(dec_output, target_tensor[di])
                dec_input = target_tensor[di]  # Teacher forcing
        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                dec_output, dec_hidden, dec_attention = self.decoder(
                    dec_input, dec_hidden, enc_outputs)
                topv, topi = dec_output.topk(1)
                dec_input = topi.squeeze().detach()  # detach from history as input

                loss += criterion(dec_output, target_tensor[di])
                if dec_input.item() == EOS_token:
                    break

        loss.backward()

        enc_optimizer.step()
        dec_optimizer.step()

        return loss.item() / target_length

    def predict(self, sentence):
        with torch.no_grad():
            input_tensor = self.tokenizer.get_tensor_from_sentence(sentence, is_input=True)
            input_length = input_tensor.size()[0]
            enc_hidden = self.encoder.initHidden(self.device)

            enc_outputs = torch.zeros(self.MAX_LENGTH, self.encoder.hidden_size, device=self.device)

            for ei in range(input_length):
                enc_output, enc_hidden = self.encoder(input_tensor[ei],
                                                        enc_hidden)
                enc_outputs[ei] += enc_output[0, 0]

            decoder_input = torch.tensor([[SOS_token]], device=self.device)  # SOS

            decoder_hidden = enc_hidden

            decoded_words = []
            decoder_attentions = torch.zeros(self.MAX_LENGTH, self.MAX_LENGTH)

            for di in range(self.MAX_LENGTH):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, enc_outputs)
                decoder_attentions[di] = decoder_attention.data
                topv, topi = decoder_output.data.topk(1)
                if topi.item() == EOS_token:
                    decoded_words.append('')
                    break
                else:
                    decoded_words.append(self.tokenizer.output_lang.index2word[topi.item()])

                decoder_input = topi.squeeze().detach()

            return decoded_words, decoder_attentions[:di + 1]



