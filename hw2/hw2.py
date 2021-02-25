#!/usr/bin/env python
# coding: utf-8

# General Imports and Helper Functions provided by Professor Holleman
# TensorFlow and tf.keras
import tensorflow as tf

# example of loading an image with the Keras API
from keras_preprocessing import image

# Helper libraries
import numpy as np
from numpy import asarray
import matplotlib.pyplot as plt

from tensorflow.keras import Input, layers

print(tf.__version__)

def quick_bar(x):
    return plt.bar(np.arange(len(x)), x)

def grouped_bar(l_values, l_group_names=None): # list of arrays, one for each group
    if l_group_names is None:
        l_group_names = ['']*len(l_values)

    frac = 0.8 # each field N has space from N-0.5 to N+0.5 to use. we'll use frac of it
    ind = np.arange(len(l_values[0]))  # the x locations for the groups

    width = frac/len(l_values)  # the width of the bars

    fig = plt.gcf()
    ax = plt.gca()
    rects = []
    for i, group_data in enumerate(l_values):
        # the spacing code needs some work.
        bar_centers = ind-0.5*frac + width/2 + i*width
        rects.append(ax.bar(bar_centers, group_data, width, label=l_group_names[i]))

    ax.set_xticks(ind)
    # ax.set_xticklabels(('G1', 'G2', 'G3', 'G4', 'G5'))
    ax.legend()


def plotyy(xdata, y1, y2, color1='tab:red', color2='tab:blue', ylabel1='', ylabel2='', xlabel=''):
    ## adapated from https://matplotlib.org/2.2.5/gallery/api/two_scales.html
    fig = plt.gcf()
    ax1 = plt.gca()

    ax1.plot(xdata, y1, color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylabel(ylabel1, color=color1)
    ax1.set_xlabel(xlabel)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel(ylabel2, color=color2)  # we already handled the x-label with ax1
    ax2.plot(xdata, y2, color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)


    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    return (ax1, ax2)


# Import CIFAR10 data
cifar10 = tf.keras.datasets.cifar10
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
input_shape = train_images.shape[1:] # All shapes except the first
print("The input shape is "+ str(input_shape))

# Show example image
idx = 62
plt.figure()
plt.imshow(train_images[idx])
plt.colorbar()
plt.title("Label = {:}".format(class_names[train_labels[idx][0]]))
plt.show()

# Show a grid of example images
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

# Create model to be trained with 5 layers
model = tf.keras.Sequential([
    # Conv 2D: 32 filters, 3x3 kernel, stride=2 (in both x,y dimensions), "same" padding
    tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="same",
        activation="relu",
        input_shape=input_shape
    ),
    # Conv 2D: 64 filters, 3x3 kernel, stride=2 (in both x,y dimensions),  "same" padding
    tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="same",
        activation="relu"
    ),
    # MaxPooling, 2x2 pooling size, 2x2 stride
    tf.keras.layers.MaxPool2D(
        pool_size=(2, 2),
        strides=(2, 2)
    ),
    # Flatten
    tf.keras.layers.Flatten(),
    # Dense (aka Fully Connected) , 1024 units.
    tf.keras.layers.Dense(1024, activation="relu"),
    # Dense (aka Fully Connected) , 10 units
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])

print("Model summary:")
model.summary()

# Train Model
epoch_num = 50

train_hist = model.fit(train_images, train_labels, epochs=epoch_num)
#Save model
model.save('./saved_models/cifar_cnn_model')

# Display training history
train_hist.history
plotyy(np.arange(epoch_num), train_hist.history['loss'], train_hist.history['accuracy'],ylabel1='Loss', ylabel2='Acc')

# Test loss and accuracy
test_loss, test_acc = model.evaluate(test_images, test_labels)

# Create second model to be trained
model2 = tf.keras.Sequential([
    # Conv 2D: 32 filters, 3x3 kernel, stride=2 (in both x,y dimensions), "same" padding
    tf.keras.layers.DepthwiseConv2D(
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="same",
        activation="relu",
        input_shape=input_shape
    ),
    tf.keras.layers.Conv2D(
        filters=32,
        kernel_size=(1, 1),
        activation="relu",
    ),
    tf.keras.layers.DepthwiseConv2D(
        kernel_size=(3, 3),
        strides=(2, 2),
        padding="same",
        activation="relu",
    ),
    # Conv 2D: 64 filters, 3x3 kernel, stride=2 (in both x,y dimensions),  "same" padding
    tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(1, 1),
        activation="relu",
    ),
    tf.keras.layers.MaxPool2D(
        pool_size=(2, 2),
        strides=(2, 2)
    ),
    # Flatten
    tf.keras.layers.Flatten(),
    # Dense (aka Fully Connected) , 1024 units.
    tf.keras.layers.Dense(1024, activation="relu"),
    # Dense (aka Fully Connected) , 10 units
    tf.keras.layers.Dense(10)
])
model2.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])
print("Model 2 summary: ")
model2.summary()

# Train model 2, display, and save
train_hist2 = model2.fit(train_images, train_labels, epochs=epoch_num)
plotyy(np.arange(epoch_num), train_hist2.history['loss'], train_hist2.history['accuracy'], ylabel1='Loss', ylabel2='Acc')
model2.save('./saved_models/cifar_cnn_model2')

# Test loss and accuracy of second model
test_loss, test_acc = model2.evaluate(test_images, test_labels)

# Test model using cifar10 image
idx = 62
ex_img = train_images[idx:idx+1]
ex_out = model.predict(ex_img)
plt.subplot(2,1,1)
plt.imshow(ex_img[0,:,:], cmap='Greys')
plt.title("Label = {:}".format(class_names[train_labels[idx][0]]))

plt.subplot(2,1,2)
plt.bar(np.arange(10), ex_out[0])
class_name = np.argmax(ex_out[0])

print("I think that this image is an " + class_names[class_name] + ".")
plt.show()

# Load personal image
img = image.load_img("./eagle.jpg", target_size=(32,32))
x = np.array([ image.img_to_array(img, dtype=np.uint8) ])

# Test model using personal image
img1_out = model.predict(x)
plt.subplot(2,1,1)
plt.imshow(x[0])
plt.title("Label = {:}".format(class_names[train_labels[7][0]]))

plt.subplot(2,1,2)
plt.bar(np.arange(10), img1_out[0])
class_name = np.argmax(img1_out[0])

print("I think that this image is an " + class_names[class_name] + ".")
plt.show()

# Test model 2 using personal image
img2_out = model2.predict(x)
plt.subplot(2,1,1)
plt.imshow(x[0])
plt.title("Label = {:}".format(class_names[train_labels[7][0]]))

plt.subplot(2,1,2)
plt.bar(np.arange(10), img2_out[0])
class_name = np.argmax(img2_out[0])

print("I think that this image is an " + class_names[class_name] + ".")
plt.show()
