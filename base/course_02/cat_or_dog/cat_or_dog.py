import os

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 4.5
# import zipfile
# local_zip='./horse_or_human.zip'
# zip_ref=zipfile.ZipFile(local_zip,'r')
# zip_ref.extractall()
# zip_ref.close()

# 获取文件路径，查看图片名
train_cats_dir = os.path.join('cats_and_dogs_filtered/train/cats')
train_dogs_dir = os.path.join('cats_and_dogs_filtered/train/dogs')
validation_cats_dir = os.path.join('cats_and_dogs_filtered/validation/cats')
validation_dogs_dir = os.path.join('cats_and_dogs_filtered/validation/dogs')
train_cats_names = os.listdir(train_cats_dir)
train_dogs_names = os.listdir(train_dogs_dir)
validation_cats_names = os.listdir(validation_cats_dir)
validation_dogs_names = os.listdir(validation_dogs_dir)
print(len(train_cats_names),
      len(train_dogs_names),
      len(validation_cats_names),
      len(validation_dogs_names))

nrows = 4
ncols = 4
pic_index = 0
# 获取画布，设置大小
fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)
# 通过循环获取图片路径
pic_index += 8
next_horse_pix = [os.path.join(train_cats_dir, fname)
                  for fname in train_cats_names[pic_index - 8:pic_index]]
next_human_pix = [os.path.join(train_dogs_dir, fname)
                  for fname in train_dogs_names[pic_index - 8:pic_index]]

# 读取图片，指定位置，画出图片
for i, img_path in enumerate(next_horse_pix + next_human_pix):
    sp = plt.subplot(nrows, ncols, i + 1)  # 从1起
    sp.axis('off')
    img = mpimg.imread(img_path)
    plt.imshow(img)
# plt.tight_layout()
# plt.show()

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu',
                           input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(2, 2),
    # tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    # tf.keras.layers.MaxPool2D(2, 2),
    # tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    # tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    # tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')  # 二分类中更好
])
model.summary()
model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['acc'])

# 加载图片,训练集图像增强，但是如果样本本身覆盖不好，增强效果并不明显
train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')
train_generator = train_datagen. \
    flow_from_directory('./cats_and_dogs_filtered/train/',
                        target_size=(150, 150),
                        batch_size=20,
                        class_mode='binary')
validation_datagen = ImageDataGenerator(rescale=1. / 255)
validation_generator = validation_datagen. \
    flow_from_directory('./cats_and_dogs_filtered/validation/',
                        target_size=(150, 150),
                        batch_size=20,
                        class_mode='binary')

# 加入验证集
history = model.fit(train_generator,
                    steps_per_epoch=100,
                    epochs=100,
                    verbose=2,
                    validation_data=validation_generator,
                    validation_steps=50)
# model.evaluate(validation_generator)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

plt.figure()
plt.plot(epochs, acc, label='acc')
plt.plot(epochs, val_acc, label='val_acc')
plt.title('Accuracy')
plt.legend()

plt.figure()
plt.plot(epochs, loss, label='loss')
plt.plot(epochs, val_loss, label='val_loss')
plt.title('Loss')
plt.legend()
plt.show()

