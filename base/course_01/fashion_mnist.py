import tensorflow as tf

# import matplotlib.pyplot as plt
mnist = tf.keras.datasets.fashion_mnist
(training_images, training_labels), \
(test_images, test_labels) = mnist.load_data()
# plt.imshow(training_images[0])
# plt.show()
print(type(training_images[0]))
print(training_labels[0])

training_images = training_images / 255.0  # 文件只读，所以不能用/=修改
test_images = test_images / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    # tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer=tf.optimizers.Adam(),
              loss=tf.losses.sparse_categorical_crossentropy,
              metrics=tf.metrics.sparse_categorical_accuracy)
model.fit(training_images, training_labels, epochs=5)
model.evaluate(test_images, test_labels)
