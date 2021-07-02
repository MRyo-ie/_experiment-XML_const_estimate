import os
import random
import torch
from tqdm import tqdm
# from livelossplot import PlotLosses
# %matplotlib inline

# from utils.Logger import showPlot
# from utils.Timer import asMinutes, timeSince

from model.seq2seq_Model import (
    Seq2Seq_batch_ptModel,
    PAD_token,
)



###  Train  ###
import time
import torch.nn as nn
from torch import optim

from torch.utils.tensorboard import SummaryWriter


class example_ExpTrain():
    def __init__(self, train_pairs, test_pairs):
        self.train_pairs = train_pairs
        self.test_pairs = test_pairs


    def exec(self, model:Seq2Seq_batch_ptModel, 
                    epochs=30, batch_size=200,
                    teacher_forcing=0.5, early_stopping=5):
        
        log_dir = model.save_m_dir

        # default `log_dir` is "runs" - we'll be more specific here
        writer = SummaryWriter(log_dir)
        # print(model.encoder)
        # writer.add_graph(model.encoder)
        # writer.add_graph(model.decoder)

        # liveloss = PlotLosses()
        optimizer = optim.Adam(
                        [p for p in model.encoder.parameters()]
                        + [p for p in model.decoder.parameters()] )

        criterion = nn.NLLLoss(ignore_index=PAD_token)

        validation_bleus = []
        
        for epc in tqdm(range(epochs)):
            total_loss = 0
            for input_batch, input_lens, target_batch, target_lens in model.generate_batch(self.train_pairs, batch_size=batch_size):
                loss = model.fit_batch(input_batch, input_lens,
                                        target_batch, target_lens,
                                        optimizer, criterion, teacher_forcing)
                total_loss += loss
                train_loss = total_loss / (len(self.train_pairs) / batch_size)
            
            model.save()

            total_bleu = 0
            for input_batch, input_lens, target_batch, target_lens in model.generate_batch(self.train_pairs, batch_size=batch_size, shuffle=False):
                loss, bleu = model.evaluate_batch(input_batch, input_lens,
                                                  target_batch, target_lens, criterion)
                total_bleu += bleu
            train_bleu = total_bleu / (len(self.train_pairs) / batch_size)
            
            total_loss = 0
            total_bleu = 0
            for input_batch, input_lens, target_batch, target_lens in model.generate_batch(self.test_pairs, batch_size=batch_size, shuffle=False):
                loss, bleu = model.evaluate_batch(input_batch, input_lens,
                                                  target_batch, target_lens, criterion)
                total_loss += loss
                total_bleu += bleu
            validation_loss = total_loss / (len(self.test_pairs) / batch_size)
            validation_bleu = total_bleu / (len(self.test_pairs) / batch_size)
            
            # liveloss.update({
            #     'loss': train_loss,
            #     'bleu': train_bleu,
            #     'val_bleu': validation_bleu
            # })
            # liveloss.draw()
            writer.add_scalar('train loss', train_loss, epc)
            writer.add_scalar('valid loss', validation_loss, epc)
            writer.add_scalar('train bleu', train_bleu, epc)
            writer.add_scalar('valid bleu', validation_bleu, epc)

            validation_bleus.append(validation_bleu)
            if max(validation_bleus[-early_stopping:]) < max(validation_bleus):
                break
        
        writer.close()
        return max(validation_bleus)







if __name__ == "__main__":
    from data.example_Data import Lang, prepareData
    from model.rnn_model.encoderRNN import (
        EncoderLSTM, 
    )
    from model.rnn_model.decoderRNN import (
        AttnDecoderLSTM1,
        AttnDecoderLSTM2,
    )
    from model.seq2seq_Model import (
        Seq2SeqTranslate_ptTokenizer,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## Data
    input_lang, output_lang, pairs = prepareData(
        'eng', 'fra', '_data_example/data', False)
    print(random.choice(pairs))
    # train / test split
    from sklearn.model_selection import train_test_split
    train_pairs, test_pairs = train_test_split(pairs, test_size=0.2)

    ## Model
    tokenizer = Seq2SeqTranslate_ptTokenizer(
                    input_lang, output_lang, device)

    ###  test  ###
    batch_size = 10
    emb_size = 8
    hid_size = 12
    MAX_LENGTH = 18

    seq2seq_test_model = Seq2Seq_batch_ptModel(
                        tokenizer, device,
                        dropout_p=0.1, max_length=MAX_LENGTH)
    test_encoder = EncoderLSTM(input_lang.n_words, emb_size, hid_size)
    test_decoder1 = AttnDecoderLSTM1(
                        emb_size, hid_size, output_lang.n_words,
                        device, max_length=MAX_LENGTH)
    
    seq2seq_test_model.load_enc_dec_models(test_encoder, test_decoder1)
    seq2seq_test_model.exec_test(train_pairs, batch_size=batch_size)

    attn_size = 9
    test_decoder2 = AttnDecoderLSTM2(
                        emb_size, hid_size, attn_size, 
                        output_lang.n_words, device).to(device)
    seq2seq_test_model.load_enc_dec_models(test_encoder, test_decoder2)
    seq2seq_test_model.exec_test(train_pairs, batch_size=batch_size)


    ###  exp 1  ###
    emb_size = 1024
    hidden_size = 1024

    encoder = EncoderLSTM(input_lang.n_words, emb_size, hidden_size).to(device)
    decoder = AttnDecoderLSTM1(emb_size, hidden_size, output_lang.n_words).to(device)
    seq2seq_model = Seq2Seq_batch_ptModel(
                        encoder, decoder, tokenizer, device,
                        dropout_p=0.1, max_length=10)

