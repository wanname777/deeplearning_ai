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
train_horse_dir = os.path.join('horse-or-human/train/horses')
train_human_dir = os.path.join('horse-or-human/train/humans')
validation_horse_dir = os.path.join('horse-or-human/validation/horses')
validation_human_dir = os.path.join('horse-or-human/validation/humans')
train_horse_names = os.listdir(train_horse_dir)
train_human_names = os.listdir(train_human_dir)
print(len(train_horse_names), len(train_human_names))

nrows = 4
ncols = 4
pic_index = 0
# 获取画布，设置大小
fig = plt.gcf()
fig.set_size_inches(ncols * 4, nrows * 4)
# 通过循环获取图片路径
pic_index += 8
next_horse_pix = [os.path.join(train_horse_dir, fname)
                  for fname in train_horse_names[pic_index - 8:pic_index]]
next_human_pix = [os.path.join(train_human_dir, fname)
                  for fname in train_human_names[pic_index - 8:pic_index]]

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
    tf.keras.layers.Dense(1, activation='sigmoid')  # 二分类中更好
])
model.summary()
model.compile(loss='binary_crossentropy',
              optimizer=RMSprop(lr=0.001),
              metrics=['acc'])

# 加载图片
train_datagen = ImageDataGenerator(rescale=1 / 255, )
train_generator = train_datagen. \
    flow_from_directory('./horse-or-human/train/',
                        target_size=(150, 150),
                        batch_size=128,
                        class_mode='binary')
validation_datagen = ImageDataGenerator(rescale=1 / 255)
validation_generator = validation_datagen. \
    flow_from_directory('./horse-or-human/validation/',
                        target_size=(150, 150),
                        batch_size=32,
                        class_mode='binary')

# 加入验证集
model.fit(train_generator,
          steps_per_epoch=8,
          epochs=15,
          verbose=1,
          validation_data=validation_generator,
          validation_steps=8)
# model.evaluate(validation_generator)
