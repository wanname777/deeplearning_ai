# 移步colab
'''
import tensorflow_datasets as tfds

imdb, info = tfds.load("imdb_reviews",
                       with_info=True,
                       as_supervised=True)

print(imdb)
print('*' * 50)
print(info)
print('*' * 50)
train_data, test_data = imdb['train'], imdb['test']
print(train_data)
training_sentences = []
training_labels = []
testing_sentences = []
testing_labels = []

'''
