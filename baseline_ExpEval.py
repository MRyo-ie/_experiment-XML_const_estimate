import random

import torch
import torch.nn as nn

from utils.Logger import showPlot, show_attention
from utils.Timer import asMinutes, timeSince

from data.example_Data import Lang, prepareData
from model.seq2seq_Model import (
    Seq2Seq_GRU_Attn_ptModel, 
    Seq2SeqTranslate_ptTokenizer,
)




###  Print (simple)  ##########################################

def eval_print_randomly(model, pairs, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = model.predict(pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        print('')



###  Print (Full)  ##########################################

def evaluate_batch_randomly(model, pairs, n=10): 
    scores = []
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_words, attentions = model.predict(pair[0])
        output_sentence = ' '.join(output_words)
        print('<', output_sentence)
        score = model.compute_bleu([[pair[1].split()]], [output_words[:-1]])
        print('bleu:', score)
        print('')

        show_attention(pair[0], output_words, attentions)
        scores.append(score)
    return scores




###  Acc  ##########################################

# # Decoderのアウトプットのテンソルから要素が最大のインデックスを返す。つまり生成文字を意味する
# def get_max_index(decoder_output):
#   results = []
#   for h in decoder_output:
#     results.append(torch.argmax(h))
#   return torch.tensor(results, device=device).view(BATCH_NUM, 1)

# # 評価用データ
# test_input_batch, test_output_batch = train2batch(test_x, test_y)
# input_tensor = torch.tensor(test_input_batch, device=device)

# predicts = []
# for i in range(len(test_input_batch)):
#   with torch.no_grad(): # 勾配計算させない
#     encoder_state = encoder(input_tensor[i])

#     # Decoderにはまず文字列生成開始を表す"_"をインプットにするので、"_"のtensorをバッチサイズ分作成
#     start_char_batch = [[char2id["_"]] for _ in range(BATCH_NUM)]
#     decoder_input_tensor = torch.tensor(start_char_batch, device=device)

#     # 変数名変換
#     decoder_hidden = encoder_state

#     # バッチ毎の結果を結合するための入れ物を定義
#     batch_tmp = torch.zeros(100,1, dtype=torch.long, device=device)
#     # print(batch_tmp.size())
#     # (100,1)

#     for _ in range(5):
#       decoder_output, decoder_hidden = decoder(decoder_input_tensor, decoder_hidden)
#       # 予測文字を取得しつつ、そのまま次のdecoderのインプットとなる
#       decoder_input_tensor = get_max_index(decoder_output.squeeze())
#       # バッチ毎の結果を予測順に結合
#       batch_tmp = torch.cat([batch_tmp, decoder_input_tensor], dim=1)

#     # 最初のbatch_tmpの0要素が先頭に残ってしまっているのでスライスして削除
#     predicts.append(batch_tmp[:,1:])

# # バッチ毎の予測結果がまとまって格納されてます。
# print(len(predicts))
# # 150
# print(predicts[0].size())
# # (100, 5)


# import pandas as pd
# id2char = {str(i) : str(i) for i in range(10)}
# id2char.update({"10":"", "11":"-", "12":""})
# row = []
# for i in range(len(test_input_batch)):
#   batch_input = test_input_batch[i]
#   batch_output = test_output_batch[i]
#   batch_predict = predicts[i]
#   for inp, output, predict in zip(batch_input, batch_output, batch_predict):
#     x = [id2char[str(idx)] for idx in inp]
#     y = [id2char[str(idx)] for idx in output]
#     p = [id2char[str(idx.item())] for idx in predict]

#     x_str = "".join(x)
#     y_str = "".join(y)
#     p_str = "".join(p)

#     judge = "O" if y_str == p_str else "X"
#     row.append([x_str, y_str, p_str, judge])
# predict_df = pd.DataFrame(row, columns=["input", "answer", "predict", "judge"])

# # 正解率を表示
# print(len(predict_df.query('judge == "O"')) / len(predict_df))
# # 0.8492
# # 間違えたデータを一部見てみる
# print(predict_df.query('judge == "X"').head(10))




if __name__ == "__main__":
    ## Experiment
    hidden_size = 256
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ## Data
    input_lang, output_lang, pairs = prepareData(
            'eng', 'fra', '_data_example/data', False)
    print(random.choice(pairs))
    ## Model
    tokenizer = Seq2SeqTranslate_ptTokenizer(input_lang, output_lang, device)
    seq2seq_model = Seq2Seq_GRU_Attn_ptModel(
                        input_lang.n_words, output_lang.n_words, hidden_size,
                        tokenizer, device, dropout_p=0.1)

    eval_print_randomly(seq2seq_model, pairs)

    # Visualizing Attention
    # output_words, attentions = evaluate(
    #         seq2seq_model, "je suis trop froid .")
    # plt.matshow(attentions.numpy())
    # plt.savefig('Attention_viz.png')

    for ib, il, ob, ol in generate_batch(train_pairs, 100):
        break
    batch_evaluation(ib, il, ob, ol, test_encoder, test_decoder1, criterion)
