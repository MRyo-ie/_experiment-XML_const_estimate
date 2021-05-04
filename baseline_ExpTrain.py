import os
import random
import torch
from tqdm import tqdm

from utils.Logger import showPlot
from utils.Timer import asMinutes, timeSince

from data.example_Data import Lang, prepareData
from model.seq2seq_Model import (
    Seq2Seq_GRU_Attn_ptModel,
    Seq2SeqTranslate_ptTokenizer,
)



###  Train  ###
import time
import torch.nn as nn
from torch import optim


class example_ExpTrain():
    def __init__(self, pairs):
        self.pairs = pairs

    def exec(self, model, n_iters,
                print_every=1000, plot_every=100, 
                learning_rate=0.01, log_dir=None):
        start = time.time()
        plot_losses = []
        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every

        encoder_optimizer = optim.SGD(model.encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.SGD(model.decoder.parameters(), lr=learning_rate)
        training_pairs = [model.tokenizer.get_tensors_from_pair(random.choice(self.pairs))
                            for i in range(n_iters)]
        criterion = nn.NLLLoss()

        for iter in tqdm(range(1, n_iters + 1)):
            training_pair = training_pairs[iter - 1]
            input_tensor = training_pair[0]
            target_tensor = training_pair[1]

            loss = model.fit(input_tensor, target_tensor,
                                encoder_optimizer, decoder_optimizer, criterion)
            print_loss_total += loss
            plot_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / print_every
                print_loss_total = 0
                print('%s (%d %d%%) %.4f' % (timeSince(start, iter / n_iters),
                                            iter, iter / n_iters * 100, print_loss_avg))

            if iter % plot_every == 0:
                plot_loss_avg = plot_loss_total / plot_every
                plot_losses.append(plot_loss_avg)
                plot_loss_total = 0

        if log_dir is not None:
            os.makedirs(log_dir, exist_ok=True)
        save_graph_parh = os.path.join(log_dir, f'loss_{n_iters}epc')
        showPlot(plot_losses, save_fname=save_graph_parh)




if __name__ == "__main__":
    hidden_size = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## Data
    input_lang, output_lang, pairs = prepareData(
        'eng', 'fra', '_data_example/data', False)
    print(random.choice(pairs))
    ## Model
    tokenizer = Seq2SeqTranslate_ptTokenizer(
                    input_lang, output_lang, device)
    seq2seq_model = Seq2Seq_GRU_Attn_ptModel(
                        input_lang.n_words, output_lang.n_words, hidden_size,
                        tokenizer, device, dropout_p=0.1)

    exp_train = example_ExpTrain(pairs)
    exp_train.exec(seq2seq_model, 75,
                    print_every=5000, plot_every=10, 
                    learning_rate=0.01, log_dir='_logs')
    # exp_train.exec(seq2seq_model, 75000,
    #                 print_every=5000, plot_every=1000, 
    #                 learning_rate=0.01, log_dir='_logs')

