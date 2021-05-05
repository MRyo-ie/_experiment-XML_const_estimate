from abc import ABCMeta, abstractmethod
import random

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from abc_ml.abc_model import ModelABC
from abc_ml.abc_tokenizer import Seq2Seq_TokenizerABC

from model.rnn_model.encoderRNN import EncoderGRU, EncoderLSTM
from model.rnn_model.decoderRNN import DecoderGRU, DecoderLSTM, AttnDecoderGRU, AttnDecoderLSTM1, AttnDecoderLSTM2


SOS_token = 0
EOS_token = 1

PAD_token = -1

"""
Pytorch チュートリアル（Seq2Seq）
・ https://torch.classcat.com/2018/05/15/pytorch-tutorial-intermediate-seq2seq-translation/
"""



####    Tokenizer    ##############################################

class Seq2SeqTranslate_ptTokenizer(Seq2Seq_TokenizerABC):
    def __init__(self, input_lang, output_lang, device, tensor_type="pt"):
        super().__init__(input_lang, output_lang, device, tensor_type)
    
    # 訓練データの準備
    # 訓練するために、各ペアについて入力 tensor (入力センテンスの単語のインデックス) とターゲット tensor (ターゲット・センテンスの単語のインデックス) が必要です。これらのベクトルを作成する一方で両者のシーケンスに EOS トークンを追加します。
    def sentence_to_indexs(self, sentence, is_input:bool):
        lang = self.input_lang if is_input else self.output_lang
        return [lang.word2index[word] for word in sentence.split(' ')]

    def sentence_to_indexs_padding(self, sentence, is_input: bool, max_length):
        indexes = sentence.split(' ')
        length = len(indexes)
        indexes = self.sentence_to_indexs(sentence, is_input)
        return indexes + [EOS_token] + [0] * (max_length - length - 1), length + 1

    def get_tensor_from_sentence(self, sentence, is_input:bool):
        indexes = self.sentence_to_indexs(sentence, is_input)
        indexes.append(EOS_token)
        return torch.tensor(indexes, dtype=torch.long, device=self.device).view(-1, 1)

    def get_tensors_from_pair(self, pair):
        input_tensor = self.get_tensor_from_sentence(pair[0], is_input=True)
        target_tensor = self.get_tensor_from_sentence(pair[1], is_input=False)
        return (input_tensor, target_tensor)



####    NN model    ##############################################

import numpy as np
from nltk.translate.bleu_score import sentence_bleu

class Seq2Seq_batch_ptModel():
    def __init__(self, tokenizer: Seq2SeqTranslate_ptTokenizer,
                    device, dropout_p=0.1, max_length=18):
        self.device = device
        self.tokenizer = tokenizer

        self.MAX_LENGTH = max_length
        self.teacher_forcing_ratio = 0.5
    
    def load_enc_dec_models(self, encoder, decoder,):
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)

    def exec_test(self, train_pairs, batch_size=10):
        for input_batch, input_lens, output_batch, output_lens in self.generate_batch(train_pairs, batch_size):
            break
        print('[Info] input_batch.shape, input_lens.shape\n     = ', input_batch.shape, input_lens.shape)

        enc_outputs, (hidden_h, hidden_c) = self.encoder(input_batch, input_lens)
        print('[Info] enc_outputs.shape, hidden_h.shape, hidden_c.shape\n     = ', enc_outputs.shape, hidden_h.shape, hidden_c.shape)

        hidden = (hidden_h.squeeze(0), hidden_c.squeeze(0))
        print('[Info] hidden[0].shape, hidden[1].shape\n     = ', hidden[0].shape, hidden[1].shape)

        dec_input = torch.tensor([SOS_token] * batch_size, device=self.device)
        dec_outputs, hidden, attn_weights = self.decoder(dec_input, hidden, enc_outputs)
        print('[Info] dec_outputs.shape, hidden[0].shape, hidden[1].shape, attn_weights.shape\n     = ', dec_outputs.shape, hidden[0].shape, hidden[1].shape, attn_weights.shape)

        criterion = nn.NLLLoss(ignore_index=PAD_token)
        loss = criterion(dec_outputs, output_batch[0])
        print('[Info] loss.item()\n     = ', loss.item())


    def generate_batch(self, pairs, batch_size=200, shuffle=True):
        random.shuffle(pairs)
        
        for i in range(len(pairs) // batch_size):
            batch_pairs = pairs[batch_size*i:batch_size*(i+1)]

            input_batch = []
            target_batch = []
            input_lens = []
            target_lens = []

            # print(len(batch_pairs))
            for input_seq, target_seq in batch_pairs:
                input_seq, input_length = self.tokenizer.sentence_to_indexs_padding(input_seq, True, self.MAX_LENGTH)
                target_seq, target_length = self.tokenizer.sentence_to_indexs_padding(target_seq, False, self.MAX_LENGTH)

                input_batch.append(input_seq)
                target_batch.append(target_seq)
                input_lens.append(input_length)
                target_lens.append(target_length)

            # print(len(input_batch), len(input_batch[2]))
            input_batch = torch.tensor(input_batch, dtype=torch.long, device=self.device)
            target_batch = torch.tensor(target_batch, dtype=torch.long, device=self.device)
            input_lens = torch.tensor(input_lens)
            target_lens = torch.tensor(target_lens)
            
            # sort
            input_lens, sorted_idxs = input_lens.sort(0, descending=True)
            input_batch = input_batch[sorted_idxs].transpose(0, 1)
            input_batch = input_batch[:input_lens.max().item()]
            
            target_batch = target_batch[sorted_idxs].transpose(0, 1)
            target_batch = target_batch[:target_lens.max().item()]
            target_lens = target_lens[sorted_idxs]
            
            yield input_batch, input_lens, target_batch, target_lens


    def fit_batch(self, input_batch, input_lens, target_batch, target_lens,
                    optimizer, criterion,  teacher_forcing_ratio=0.5):
        loss = 0
        optimizer.zero_grad()

        batch_size = input_batch.shape[1]
        target_length = target_lens.max().item()

        enc_outputs, enc_hidden = self.encoder(input_batch, input_lens)  # (s, b, 2h), ((1, b, h), (1, b, h))
        
        dec_input = torch.tensor([[SOS_token] * batch_size], device=self.device)  # (1, b)
        dec_inputs = torch.cat([dec_input, target_batch], dim=0)  # (1,b), (n,b) -> (n+1, b)
        dec_hidden = (enc_hidden[0].squeeze(0), enc_hidden[1].squeeze(0))
        
        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
        
            for di in range(target_length):
                dec_output, dec_hidden, attention = self.decoder(
                    dec_inputs[di], dec_hidden, enc_outputs)

                loss += criterion(dec_output, dec_inputs[di+1])
        else:
            dec_input = dec_inputs[0]
            for di in range(target_length):
                dec_output, dec_hidden, attention = self.decoder(
                    dec_input, dec_hidden, enc_outputs)

                loss += criterion(dec_output, dec_inputs[di+1])

                _, topi = dec_output.topk(1)  # (b,odim) -> (b,1)
                dec_input = topi.squeeze(1).detach() 

        loss.backward()
        optimizer.step()

        return loss.item() / target_length


    def evaluate_batch(self, input_batch, input_lens,
                            target_batch, target_lens, criterion):
        with torch.no_grad():
            batch_size = input_batch.shape[1]
            target_length = target_lens.max().item()
            target_batch = target_batch[:target_length]

            loss = 0
            
            enc_outputs, enc_hidden = self.encoder(input_batch, input_lens)  # (s, b, 2h), ((1, b, h), (1, b, h))
            dec_input = torch.tensor([SOS_token] * batch_size, device=self.device)  # (b)
            dec_hidden = (enc_hidden[0].squeeze(0), enc_hidden[1].squeeze(0))

            o_lang = self.tokenizer.output_lang
            decoded_outputs = torch.zeros(target_length, batch_size, o_lang.n_words, device=self.device)
            decoded_words = torch.zeros(batch_size, target_length, device=self.device)
            
            for di in range(target_length):
                dec_output, dec_hidden, _ = self.decoder(
                    dec_input, dec_hidden, enc_outputs)  # (b,odim), ((b,h),(b,h)), (b,il)        
                decoded_outputs[di] = dec_output
                
                loss += criterion(dec_output, target_batch[di])
            
                _, topi = dec_output.topk(1)  # (b,odim) -> (b,1)
                decoded_words[:, di] = topi[:, 0]  # (b)
                dec_input = topi.squeeze(1)
            
            bleu = 0
            for bi in range(batch_size):
                try:
                    end_idx = decoded_words[bi, :].tolist().index(EOS_token)
                except:
                    end_idx = target_length
                score = self.compute_bleu(
                    [[[o_lang.index2word[i] for i in target_batch[:, bi].tolist() if i > 2]]],
                    [[o_lang.index2word[j] for j in decoded_words[bi, :].tolist()[:end_idx]]]
                )
                bleu += score

            return loss.item() / target_length, bleu / float(batch_size)

    def compute_bleu(self, trues, preds):
        return np.mean([sentence_bleu(gt, p) for gt, p in zip(trues, preds)])


    def predict(self, sentence, max_length=18):
        with torch.no_grad():
            input_indxs, input_length = self.tokenizer.sentence_to_indexs_padding(sentence, True,  self.MAX_LENGTH)
            input_batch = torch.tensor([input_indxs], dtype=torch.long, device=self.device)  # (1, s)
            input_length = torch.tensor([input_length])  # (1)
            
            encoder_outputs, encoder_hidden = self.encoder(input_batch.transpose(0, 1), input_length)

            decoder_input = torch.tensor([SOS_token], device=self.device)  # (1)
            decoder_hidden = (encoder_hidden[0].squeeze(0), encoder_hidden[1].squeeze(0))

            decoded_words = []
            attentions = []
            o_lang = self.tokenizer.output_lang
            for di in range(max_length):
                decoder_output, decoder_hidden, attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)  # (1,odim), ((1,h),(1,h)), (l,1)
                attentions.append(attention)
                _, topi = decoder_output.topk(1)  # (1, 1)
                if topi.item() == EOS_token:
                    decoded_words.append('<EOS>')
                    break
                else:
                    decoded_words.append(o_lang.index2word[topi.item()])

                decoder_input = topi[0]
            
            attentions = torch.cat(attentions, dim=0)  # (l, n)
            
            return decoded_words, attentions.squeeze(0).cpu().numpy()





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

            dec_input = torch.tensor([[SOS_token]], device=self.device)  # SOS

            dec_hidden = enc_hidden

            decoded_words = []
            decoder_attentions = torch.zeros(self.MAX_LENGTH, self.MAX_LENGTH)

            for di in range(self.MAX_LENGTH):
                dec_output, dec_hidden, decoder_attention = self.decoder(
                    dec_input, dec_hidden, enc_outputs)
                decoder_attentions[di] = decoder_attention.data
                topv, topi = dec_output.data.topk(1)
                if topi.item() == EOS_token:
                    decoded_words.append('')
                    break
                else:
                    decoded_words.append(self.tokenizer.output_lang.index2word[topi.item()])

                dec_input = topi.squeeze().detach()

            return decoded_words, decoder_attentions[:di + 1]



