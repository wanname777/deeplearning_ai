import matplotlib.pyplot as plt
import tensorflow as tf

# 3.6
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), \
(test_images, test_labels) = mnist.load_data()
# plt.imshow(training_images[0])
# plt.show()
print(training_images.shape, test_images.shape)
print(type(training_images[0]))
print(training_labels[0])

training_images = training_images.reshape(60000, 28, 28, 1)
test_images = test_images.reshape(10000, 28, 28, 1)
training_images = training_images / 255.0  # 文件只读，所以不能用/=修改
test_images = test_images / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3),
                           activation=tf.nn.relu,
                           input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3),
                           activation=tf.nn.relu),
    tf.keras.layers.MaxPool2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    # tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer=tf.optimizers.Adam(),
              loss=tf.losses.sparse_categorical_crossentropy,
              metrics=tf.metrics.sparse_categorical_accuracy)
model.summary()
# early_stopping=tf.keras.callbacks.EarlyStopping(monitor='loss',patience=0.2)
model.fit(training_images,
          training_labels,
          epochs=5,
          # callbacks=early_stopping,
          verbose=2)
model.evaluate(test_images, test_labels)

# 挑几个画图
f, axarr = plt.subplots(3, 5)
FIRST_IMAGE = 0
SECOND_IMAGE = 23
THIRD_IMAGE = 28
CONVOLUTION_NUMBER = 1

axarr[0, 0].imshow(test_images[FIRST_IMAGE], cmap='inferno')
axarr[1, 0].imshow(test_images[SECOND_IMAGE], cmap='inferno')
axarr[2, 0].imshow(test_images[THIRD_IMAGE], cmap='inferno')

layer_outputs = [layer.output for layer in model.layers]
activation_model = tf.keras.models.Model(
    inputs=model.inputs, outputs=layer_outputs)
for x in range(1, 5):
    f1 = activation_model.predict(
        test_images[FIRST_IMAGE].reshape(1, 28, 28, 1))[x - 1]
    axarr[0, x].imshow(f1[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
    f2 = activation_model.predict(
        test_images[SECOND_IMAGE].reshape(1, 28, 28, 1))[x - 1]
    axarr[1, x].imshow(f2[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')
    f3 = activation_model.predict(
        test_images[THIRD_IMAGE].reshape(1, 28, 28, 1))[x - 1]
    axarr[2, x].imshow(f1[0, :, :, CONVOLUTION_NUMBER], cmap='inferno')

plt.grid(False)
plt.show()
