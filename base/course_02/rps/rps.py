import os

import matplotlib.pyplot as plt
import tensorflow as tf
from keras_preprocessing.image import ImageDataGenerator

FILE_PATH = '/base/course_02/rps/Rock-Paper-Scissors/'

train_dir = os.path.join(FILE_PATH, 'train/')
validation_dir = os.path.join(FILE_PATH, 'test/')

train_rock_dir = os.path.join(train_dir, 'rock')
train_paper_dir = os.path.join(train_dir, 'paper')
train_scissors_dir = os.path.join(train_dir, 'scissors')

validation_rock_dir = os.path.join(validation_dir, 'rock')
validation_paper_dir = os.path.join(validation_dir, 'paper')
validation_scissors_dir = os.path.join(validation_dir, 'scissors')

train_rock_names = os.listdir(train_rock_dir)
train_paper_names = os.listdir(train_paper_dir)
train_scissors_names = os.listdir(train_scissors_dir)

validation_rock_names = os.listdir(validation_rock_dir)
validation_paper_names = os.listdir(validation_paper_dir)
validation_scissors_names = os.listdir(validation_scissors_dir)

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
train_generator = train_datagen. \
    flow_from_directory(train_dir,
                        target_size=(150, 150),
                        # batch_size=20,
                        class_mode='categorical')
validation_datagen = ImageDataGenerator(rescale=1. / 255)
validation_generator = validation_datagen. \
    flow_from_directory(validation_dir,
                        target_size=(150, 150),
                        # batch_size=20,
                        class_mode='categorical')
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu',
                           input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    # tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    # tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])
model.summary()
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
EPOCHS = 10
history = model.fit(train_generator,
                    # steps_per_epoch=100,
                    epochs=EPOCHS,
                    verbose=2,
                    validation_data=validation_generator,
                    # validation_steps=50,
                    )
model.save('rps.h5')
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

plt.figure()
plt.subplot(121)
plt.plot(epochs, acc, label='accuracy')
plt.plot(epochs, val_acc, label='val_accuracy')
plt.title('Accuracy')
plt.legend()

plt.subplot(122)
plt.plot(epochs, loss, label='loss')
plt.plot(epochs, val_loss, label='val_loss')
plt.title('Loss')
plt.legend()
plt.tight_layout()
plt.show()
