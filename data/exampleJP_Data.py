import os.path as osp
import re
import random
import unicodedata


# SOS_token = 0
# EOS_token = 1

MAX_LENGTH = 18


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"}
        self.n_words = 3  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1



eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)



def prepareData(lang1, lang2, root_dir='small_parallel_enja', reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, root_dir, reverse)
    print("Read %s sentence pairs" % len(pairs))
    print("Counting words...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted words:")
    print(input_lang.name, input_lang.n_words)
    print(output_lang.name, output_lang.n_words)
    return input_lang, output_lang, pairs


def readLangs(lang1, lang2, root_dir, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    ds_f_basename = osp.join(root_dir, 'train.')
    lines1 = open(ds_f_basename+lang1, encoding='utf-8').read().strip().split('\n')
    lines2 = open(ds_f_basename+lang2, encoding='utf-8').read().strip().split('\n')

    # Split every line into pairs
    pairs = [[s1, s2] for (s1, s2) in zip(lines1, lines2)]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs


# # Turn a Unicode string to plain ASCII, thanks to
# # http://stackoverflow.com/a/518232/2809427
# def unicodeToAscii(s):
#     return ''.join(
#         c for c in unicodedata.normalize('NFD', s)
#         if unicodedata.category(c) != 'Mn'
#     )

# # Lowercase, trim, and remove non-letter characters
# def normalizeString(s):
#     s = unicodeToAscii(s.lower().strip())
#     s = re.sub(r"([.!?])", r" \1", s)
#     s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
#     return s


