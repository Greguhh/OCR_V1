import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# # testing, training data
# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

# x_train = tf.keras.utils.normalize(x_train, axis = 1)
# x_test = tf.keras.utils.normalize(x_test, axis = 1)


# # below is training the initial model 
# # (saved in a file to load after so you dont have to train every time)

# model = tf.keras.models.Sequential()
# model.add(tf.keras.layers.Flatten(input_shape = (28, 28)))
# model.add(tf.keras.layers.Dense(128, activation = 'relu'))
# model.add(tf.keras.layers.Dense(128, activation = 'relu'))
# # softmax will make sure that each output neuron value adds up to 1, so the value on an individual 
# # output is the confidence that it is that value
# model.add(tf.keras.layers.Dense(10, activation = 'softmax'))

# model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

# model.fit(x_train, y_train, epochs = 3)

# model.save('handwritten_new_actually.keras')

model = tf.keras.models.load_model('handwritten_new_actually.keras')

# loss, accuracy = model.evaluate(x_test, y_test)

# print(loss)
# print(accuracy)

image_number = 1
while os.path.isfile(f"digits/digit{image_number}.png"):
    try:
        img = cv2.imread(f"digits/digit{image_number}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"This digit is probably a {np.argmax(prediction)}")
        plt.imshow(img[0], cmap = plt.cm.binary)
        plt.show()
    except:
        print("Error!")
    finally:
        image_number += 1