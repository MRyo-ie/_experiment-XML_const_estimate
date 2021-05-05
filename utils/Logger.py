import matplotlib.pyplot as plt
# plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np

# Attentionの可視化で日本語フォントを使う
plt.rcParams['font.family'] = 'Ume Gothic O5'
plt.rcParams['font.size'] = 10


def showPlot(points, save_fname=None):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

    if save_fname is not None:
        plt.savefig(f'{save_fname}.png')




def show_attention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    input_words = input_sentence.split(' ')
    
    fig, ax = plt.subplots()
    cax = ax.matshow(attentions[:, :len(output_words)], cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_words + ['<EOS>'], rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()


