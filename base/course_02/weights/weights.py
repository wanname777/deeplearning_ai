import os

import matplotlib.pyplot as plt
from tensorflow.keras import Model
from tensorflow.keras import layers
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 加载已有的神经网络
local_weights_file = 'inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
pre_trained_model = InceptionV3(input_shape=(150, 150, 3),
                                include_top=False,
                                weights=None)
pre_trained_model.load_weights(local_weights_file)

# 关闭了训练层？
for layer in pre_trained_model.layers:
    layer.trainable = False

# 获取mixed7作为输出层
last_layer = pre_trained_model.get_layer('mixed7')
# pre_trained_model.summary()
print(last_layer.output_shape)
last_output = last_layer.output

# 连接DNN
x = layers.Flatten()(last_output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(1, activation='sigmoid')(x)

# Model API
model = Model(pre_trained_model.input, x)
model.compile(optimizer=RMSprop(lr=0.0001),
              loss='binary_crossentropy',
              metrics=['acc'])

# 加载猫狗数据集
FILE_PATH = '/base/course_02/cat_or_dog/cats_and_dogs_filtered/'
train_dir = os.path.join(FILE_PATH, 'train/')
validation_dir = os.path.join(FILE_PATH, 'validation/')
train_cats_dir = os.path.join(train_dir, 'cats')
train_dogs_dir = os.path.join(train_dir, 'dogs')
validation_cats_dir = os.path.join(validation_dir, 'cats')
validation_dogs_dir = os.path.join(validation_dir, 'dogs')
train_cats_names = os.listdir(train_cats_dir)
train_dogs_names = os.listdir(train_dogs_dir)
validation_cats_names = os.listdir(validation_cats_dir)
validation_dogs_names = os.listdir(validation_dogs_dir)

# 图像生成器
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
                        batch_size=20,
                        class_mode='binary')
validation_datagen = ImageDataGenerator(rescale=1. / 255)
validation_generator = validation_datagen. \
    flow_from_directory(validation_dir,
                        target_size=(150, 150),
                        batch_size=20,
                        class_mode='binary')
# 训练
EPOCHS = 5
history = model.fit(train_generator,
                    steps_per_epoch=100,
                    epochs=EPOCHS,
                    verbose=2,
                    validation_data=validation_generator,
                    validation_steps=50)

# acc、loss画图
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(len(acc))

plt.figure()
plt.subplot(121)
plt.plot(epochs, acc, label='acc')
plt.plot(epochs, val_acc, label='val_acc')
plt.title('Accuracy')
plt.legend()

plt.subplot(122)
plt.plot(epochs, loss, label='loss')
plt.plot(epochs, val_loss, label='val_loss')
plt.title('Loss')
plt.legend()
plt.tight_layout()
plt.show()

