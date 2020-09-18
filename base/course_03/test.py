import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

print(tf.__version__)
sentences = [
    'I love my dog',
    'I love my cat',
    'You love my dog!',
    'Do you think my dog is amazing?'
]

# out_of_vocabulary_token
tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(word_index)
sequences = tokenizer.texts_to_sequences(sentences)
print(sequences)

# padding_post为后填充,post means after/subsequent to
# truncating_post为后截断，这适用于maxlen生效的时候
padded = pad_sequences(sequences,
                       padding='post',
                       truncating='post',
                       maxlen=5)
print(padded)

test_data = [
    'i really love my dog',
    'my dog loves my manatee'
]
test_seq = tokenizer.texts_to_sequences(test_data)
print(test_seq)

padded = pad_sequences(test_seq, maxlen=10)
print(padded)

