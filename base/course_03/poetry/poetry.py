import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()

data = open('lyrics.txt', encoding='utf-8').read()

corpus = data.lower().split("\n")

tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1

# 获取每句话的不同长度的枚举
input_sequences = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    # 由于range的特性，这里不用特殊处理(1,0)、(1,1)的情况
    for i in range(1, len(token_list)):
        n_gram_sequences = token_list[:i + 1]
        input_sequences.append(n_gram_sequences)

max_sequences_len = max([len(x) for x in input_sequences])
input_sequences = np.array(pad_sequences(input_sequences))

# 划分数据集
xs, labels = input_sequences[:, :-1], input_sequences[:, -1]
# ys似one_hot编码
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

model = Sequential()
# -1是因为输入为xs
model.add(Embedding(total_words, 100, input_length=max_sequences_len - 1))
model.add(Bidirectional(LSTM(150)))
# total_words决定了要用ys而不是labels
model.add(Dense(total_words, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.01),
              metrics=['accuracy'])

# 同上，ys是因为最后Dense为多分类输出
num_epochs = 100
history = model.fit(xs, ys, epochs=num_epochs, verbose=1)


def plot_graphs(history, string):
    plt.plot(history.history[string])
    # plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    # plt.legend([string, 'val_' + string])
    plt.show()


plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

seed_text = "I've got a bad feeling about this"
next_word = 100
for i in range(next_word):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    # pad_sequences必须是二维，-1仍然是只作为x
    token_list = pad_sequences([token_list], maxlen=max_sequences_len - 1)
    # predict返回各种答案的概率
    predicted = np.argmax(model.predict(token_list), axis=-1)
    output_word = tokenizer.index_word[predicted[0]]
    seed_text += " " + output_word

print(seed_text)

