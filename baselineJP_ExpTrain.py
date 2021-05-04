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




def generate_batch(pairs, batch_size=200, shuffle=True):
    random.shuffle(pairs)
    
    for i in range(len(pairs) // batch_size):
        batch_pairs = pairs[batch_size*i:batch_size*(i+1)]

        input_batch = []
        target_batch = []
        input_lens = []
        target_lens = []
        for input_seq, target_seq in batch_pairs:
            input_seq, input_length = tensorFromSentence(input_lang, input_seq)
            target_seq, target_length = tensorFromSentence(output_lang, target_seq)

            input_batch.append(input_seq)
            target_batch.append(target_seq)
            input_lens.append(input_length)
            target_lens.append(target_length)

        input_batch = torch.tensor(input_batch, dtype=torch.long, device=device)
        target_batch = torch.tensor(target_batch, dtype=torch.long, device=device)
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

