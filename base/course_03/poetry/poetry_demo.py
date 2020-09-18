import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
data = '''
In the town of Athy one Jeremy Lanigan
battered away till he hadn't a pound
his father he died and made him a man again
left a farm with ten acres of ground
he gave a grand party for friends a relations
who did not forget him when come to the will
and if you'll but listen I'll make you're eyes glisten
of rows and ructions at Lanigan's Ball

CHORUS
six long months I spent in Dub-i-lin
six long months doing nothin' at all

I stepped out I stepped in again
learning to dance for Lanigan's Ball

Myself to be sure got free invitaions
for all the nice boys and girls I did ask
in less than 10 minutes the friends and relations
were dancing as merry as bee 'round a cask
There was lashing of punch and wine for the ladies
potatoes and cakes there was bacon a tay
there were the O'Shaughnessys, Murphys, Walshes, O'Gradys
courtin' the girls and dancing away

they were doing all kinds of nonsensical polkas
all 'round the room in a whirly gig
but Julia and I soon banished their nonsense
and tipped them a twist of a real Irish jig
Oh how that girl got mad on me
and danced till you'd think the ceilings would fall
for I spent three weeks at Brook's academy
learning to dance for Lanigan's Ball CHORUS

The boys were all merry the girls were all hearty
dancing away in couples and groups
till an accident happened young Terrance McCarthy
put his right leg through Miss Finerty's hoops
The creature she fainted and cried 'melia murder'
cried for her brothers and gathered them all
Carmody swore that he'd go no further
till he'd have satisfaction at Lanigan's Ball

In the midst of the row Miss Kerrigan fainted
her cheeks at the same time as red as a rose
some of the boys decreed she was painted
she took a wee drop too much I suppose
Her sweetheart Ned Morgan all powerful and able
when he saw his fair colleen stretched out by the wall
he tore the left leg from under the table
and smashed all the dishes at Lanigan's Ball CHORUS

Boy oh Boys tis then there was ructions
myself got a kick from big Phelam McHugh
but soon I replied to this kind introduction
and kicked up a terrible hullaballoo
old Casey the piper was near being strangled
they squeezed up his pipes bellows chanters and all
the girls in their ribbons they all got entangled
and that put an end to Lanigan's Ball CHORUS
'''
corpus = data.lower().split('\n')

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
model.add(Embedding(total_words, 64, input_length=max_sequences_len - 1))
model.add(Bidirectional(LSTM(20)))
# total_words决定了要用ys而不是labels
model.add(Dense(total_words, activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# 同上，ys是因为最后Dense为多分类输出
num_epochs = 500
history = model.fit(xs, ys, epochs=num_epochs, verbose=0)


def plot_graphs(history, string):
    plt.plot(history.history[string])
    # plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    # plt.legend([string, 'val_' + string])
    plt.show()


plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

seed_text = "Laurence went to dublin"
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

