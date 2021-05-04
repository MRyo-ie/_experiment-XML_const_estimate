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




