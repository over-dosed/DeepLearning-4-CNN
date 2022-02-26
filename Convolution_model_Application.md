# Convolutional Neural Networks: Application

Welcome to Course 4's second assignment! In this notebook, you will:

- Create a mood classifer using the TF Keras Sequential API
- Build a ConvNet to identify sign language digits using the TF Keras Functional API

**After this assignment you will be able to:**

- Build and train a ConvNet in TensorFlow for a __binary__ classification problem
- Build and train a ConvNet in TensorFlow for a __multiclass__ classification problem
- Explain different use cases for the Sequential and Functional APIs

To complete this assignment, you should already be familiar with TensorFlow. If you are not, please refer back to the **TensorFlow Tutorial** of the third week of Course 2 ("**Improving deep neural networks**").

## Table of Contents

- [1 - Packages](#1)
    - [1.1 - Load the Data and Split the Data into Train/Test Sets](#1-1)
- [2 - Layers in TF Keras](#2)
- [3 - The Sequential API](#3)
    - [3.1 - Create the Sequential Model](#3-1)
        - [Exercise 1 - happyModel](#ex-1)
    - [3.2 - Train and Evaluate the Model](#3-2)
- [4 - The Functional API](#4)
    - [4.1 - Load the SIGNS Dataset](#4-1)
    - [4.2 - Split the Data into Train/Test Sets](#4-2)
    - [4.3 - Forward Propagation](#4-3)
        - [Exercise 2 - convolutional_model](#ex-2)
    - [4.4 - Train the Model](#4-4)
- [5 - History Object](#5)
- [6 - Bibliography](#6)

<a name='1'></a>
## 1 - Packages

As usual, begin by loading in the packages.


```python
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import scipy
from PIL import Image
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.python.framework import ops
from cnn_utils import *
from test_utils import summary, comparator

%matplotlib inline
np.random.seed(1)
```

<a name='1-1'></a>
### 1.1 - Load the Data and Split the Data into Train/Test Sets

You'll be using the Happy House dataset for this part of the assignment, which contains images of peoples' faces. Your task will be to build a ConvNet that determines whether the people in the images are smiling or not -- because they only get to enter the house if they're smiling!  


```python
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_happy_dataset()

# Normalize image vectors
X_train = X_train_orig/255.
X_test = X_test_orig/255.

# Reshape
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
```

    number of training examples = 600
    number of test examples = 150
    X_train shape: (600, 64, 64, 3)
    Y_train shape: (600, 1)
    X_test shape: (150, 64, 64, 3)
    Y_test shape: (150, 1)


You can display the images contained in the dataset. Images are **64x64** pixels in RGB format (3 channels).


```python
index = 124
plt.imshow(X_train_orig[index]) #display sample training image
plt.show()
```


![png](output_7_0.png)


<a name='2'></a>
## 2 - Layers in TF Keras 

In the previous assignment, you created layers manually in numpy. In TF Keras, you don't have to write code directly to create layers. Rather, TF Keras has pre-defined layers you can use. 

When you create a layer in TF Keras, you are creating a function that takes some input and transforms it into an output you can reuse later. Nice and easy! 

<a name='3'></a>
## 3 - The Sequential API

In the previous assignment, you built helper functions using `numpy` to understand the mechanics behind convolutional neural networks. Most practical applications of deep learning today are built using programming frameworks, which have many built-in functions you can simply call. Keras is a high-level abstraction built on top of TensorFlow, which allows for even more simplified and optimized model creation and training. 

For the first part of this assignment, you'll create a model using TF Keras' Sequential API, which allows you to build layer by layer, and is ideal for building models where each layer has **exactly one** input tensor and **one** output tensor. 

As you'll see, using the Sequential API is simple and straightforward, but is only appropriate for simpler, more straightforward tasks. Later in this notebook you'll spend some time building with a more flexible, powerful alternative: the Functional API. 
 

<a name='3-1'></a>
### 3.1 - Create the Sequential Model

As mentioned earlier, the TensorFlow Keras Sequential API can be used to build simple models with layer operations that proceed in a sequential order. 

You can also add layers incrementally to a Sequential model with the `.add()` method, or remove them using the `.pop()` method, much like you would in a regular Python list.

Actually, you can think of a Sequential model as behaving like a list of layers. Like Python lists, Sequential layers are ordered, and the order in which they are specified matters.  If your model is non-linear or contains layers with multiple inputs or outputs, a Sequential model wouldn't be the right choice!

For any layer construction in Keras, you'll need to specify the input shape in advance. This is because in Keras, the shape of the weights is based on the shape of the inputs. The weights are only created when the model first sees some input data. Sequential models can be created by passing a list of layers to the Sequential constructor, like you will do in the next assignment.

<a name='ex-1'></a>
### Exercise 1 - happyModel

Implement the `happyModel` function below to build the following model: `ZEROPAD2D -> CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> FLATTEN -> DENSE`. Take help from [tf.keras.layers](https://www.tensorflow.org/api_docs/python/tf/keras/layers) 

Also, plug in the following parameters for all the steps:

 - [ZeroPadding2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/ZeroPadding2D): padding 3, input shape 64 x 64 x 3
 - [Conv2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D): Use 32 7x7 filters, stride 1
 - [BatchNormalization](https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization): for axis 3
 - [ReLU](https://www.tensorflow.org/api_docs/python/tf/keras/layers/ReLU)
 - [MaxPool2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D): Using default parameters
 - [Flatten](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten) the previous output.
 - Fully-connected ([Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense)) layer: Apply a fully connected layer with 1 neuron and a sigmoid activation. 
 
 
 **Hint:**
 
 Use **tfl** as shorthand for **tensorflow.keras.layers**


```python
# GRADED FUNCTION: happyModel

def happyModel():
    """
    Implements the forward propagation for the binary classification model:
    ZEROPAD2D -> CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> FLATTEN -> DENSE
    
    Note that for simplicity and grading purposes, you'll hard-code all the values
    such as the stride and kernel (filter) sizes. 
    Normally, functions should take these values as function parameters.
    
    Arguments:
    None

    Returns:
    model -- TF Keras model (object containing the information for the entire training process) 
    """
    model = tf.keras.Sequential([
            tfl.InputLayer(input_shape=(64,64,3)),
            ## ZeroPadding2D with padding 3, input shape of 64 x 64 x 3
            tfl.ZeroPadding2D(padding=3,data_format="channels_last"),
            ## Conv2D with 32 7x7 filters and stride of 1
            tfl.Conv2D(32,7),
            ## BatchNormalization for axis 3
            tfl.BatchNormalization(axis = 3),
            ## ReLU
            tfl.ReLU(),
            ## Max Pooling 2D with default parameters
            tfl.MaxPool2D(),
            ## Flatten layer
            tfl.Flatten(),
            ## Dense layer with 1 unit for output & 'sigmoid' activation
            tfl.Dense(units=1,activation="sigmoid"),
            # YOUR CODE STARTS HERE
            
            
            # YOUR CODE ENDS HERE
        ])

    return model
```


```python
happy_model = happyModel()
print(happy_model.summary())
```

    Model: "sequential_18"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    zero_padding2d_17 (ZeroPaddi (None, 70, 70, 3)         0         
    _________________________________________________________________
    conv2d_16 (Conv2D)           (None, 64, 64, 32)        4736      
    _________________________________________________________________
    batch_normalization_15 (Batc (None, 64, 64, 32)        128       
    _________________________________________________________________
    re_lu_14 (ReLU)              (None, 64, 64, 32)        0         
    _________________________________________________________________
    max_pooling2d_13 (MaxPooling (None, 32, 32, 32)        0         
    _________________________________________________________________
    flatten_10 (Flatten)         (None, 32768)             0         
    _________________________________________________________________
    dense_12 (Dense)             (None, 1)                 32769     
    =================================================================
    Total params: 37,633
    Trainable params: 37,569
    Non-trainable params: 64
    _________________________________________________________________
    None



```python
happy_model = happyModel()
# Print a summary for each layer
for layer in summary(happy_model):
    print(layer)
    
output = [['ZeroPadding2D', (None, 70, 70, 3), 0, ((3, 3), (3, 3))],
            ['Conv2D', (None, 64, 64, 32), 4736, 'valid', 'linear', 'GlorotUniform'],
            ['BatchNormalization', (None, 64, 64, 32), 128],
            ['ReLU', (None, 64, 64, 32), 0],
            ['MaxPooling2D', (None, 32, 32, 32), 0, (2, 2), (2, 2), 'valid'],
            ['Flatten', (None, 32768), 0],
            ['Dense', (None, 1), 32769, 'sigmoid']]
    
comparator(summary(happy_model), output)
```

    ['ZeroPadding2D', (None, 70, 70, 3), 0, ((3, 3), (3, 3))]
    ['Conv2D', (None, 64, 64, 32), 4736, 'valid', 'linear', 'GlorotUniform']
    ['BatchNormalization', (None, 64, 64, 32), 128]
    ['ReLU', (None, 64, 64, 32), 0]
    ['MaxPooling2D', (None, 32, 32, 32), 0, (2, 2), (2, 2), 'valid']
    ['Flatten', (None, 32768), 0]
    ['Dense', (None, 1), 32769, 'sigmoid']
    [32mAll tests passed![0m


Now that your model is created, you can compile it for training with an optimizer and loss of your choice. When the string `accuracy` is specified as a metric, the type of accuracy used will be automatically converted based on the loss function used. This is one of the many optimizations built into TensorFlow that make your life easier! If you'd like to read more on how the compiler operates, check the docs [here](https://www.tensorflow.org/api_docs/python/tf/keras/Model#compile).


```python
happy_model.compile(optimizer='adam',
                   loss='binary_crossentropy',
                   metrics=['accuracy'])
```

It's time to check your model's parameters with the `.summary()` method. This will display the types of layers you have, the shape of the outputs, and how many parameters are in each layer. 


```python
happy_model.summary()
```

    Model: "sequential_19"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    zero_padding2d_18 (ZeroPaddi (None, 70, 70, 3)         0         
    _________________________________________________________________
    conv2d_17 (Conv2D)           (None, 64, 64, 32)        4736      
    _________________________________________________________________
    batch_normalization_16 (Batc (None, 64, 64, 32)        128       
    _________________________________________________________________
    re_lu_15 (ReLU)              (None, 64, 64, 32)        0         
    _________________________________________________________________
    max_pooling2d_14 (MaxPooling (None, 32, 32, 32)        0         
    _________________________________________________________________
    flatten_11 (Flatten)         (None, 32768)             0         
    _________________________________________________________________
    dense_13 (Dense)             (None, 1)                 32769     
    =================================================================
    Total params: 37,633
    Trainable params: 37,569
    Non-trainable params: 64
    _________________________________________________________________


<a name='3-2'></a>
### 3.2 - Train and Evaluate the Model

After creating the model, compiling it with your choice of optimizer and loss function, and doing a sanity check on its contents, you are now ready to build! 

Simply call `.fit()` to train. That's it! No need for mini-batching, saving, or complex backpropagation computations. That's all been done for you, as you're using a TensorFlow dataset with the batches specified already. You do have the option to specify epoch number or minibatch size if you like (for example, in the case of an un-batched dataset).


```python
happy_model.fit(X_train, Y_train, epochs=10, batch_size=16)
```

    Epoch 1/10
    38/38 [==============================] - 4s 100ms/step - loss: 1.3510 - accuracy: 0.6467
    Epoch 2/10
    38/38 [==============================] - 4s 95ms/step - loss: 0.2527 - accuracy: 0.8967
    Epoch 3/10
    38/38 [==============================] - 4s 100ms/step - loss: 0.1796 - accuracy: 0.9300
    Epoch 4/10
    38/38 [==============================] - 4s 95ms/step - loss: 0.1238 - accuracy: 0.9533
    Epoch 5/10
    38/38 [==============================] - 4s 95ms/step - loss: 0.0959 - accuracy: 0.9733
    Epoch 6/10
    38/38 [==============================] - 4s 95ms/step - loss: 0.0863 - accuracy: 0.9733
    Epoch 7/10
    38/38 [==============================] - 4s 95ms/step - loss: 0.1172 - accuracy: 0.9683
    Epoch 8/10
    38/38 [==============================] - 4s 95ms/step - loss: 0.0927 - accuracy: 0.9683
    Epoch 9/10
    38/38 [==============================] - 4s 95ms/step - loss: 0.0763 - accuracy: 0.9683
    Epoch 10/10
    38/38 [==============================] - 4s 95ms/step - loss: 0.0804 - accuracy: 0.9683





    <tensorflow.python.keras.callbacks.History at 0x7ff74649f750>



After that completes, just use `.evaluate()` to evaluate against your test set. This function will print the value of the loss function and the performance metrics specified during the compilation of the model. In this case, the `binary_crossentropy` and the `accuracy` respectively.


```python
happy_model.evaluate(X_test, Y_test)
```

    5/5 [==============================] - 0s 35ms/step - loss: 0.6596 - accuracy: 0.7133





    [0.6596176028251648, 0.7133333086967468]



Easy, right? But what if you need to build a model with shared layers, branches, or multiple inputs and outputs? This is where Sequential, with its beautifully simple yet limited functionality, won't be able to help you. 

Next up: Enter the Functional API, your slightly more complex, highly flexible friend.  

<a name='4'></a>
## 4 - The Functional API

Welcome to the second half of the assignment, where you'll use Keras' flexible [Functional API](https://www.tensorflow.org/guide/keras/functional) to build a ConvNet that can differentiate between 6 sign language digits. 

The Functional API can handle models with non-linear topology, shared layers, as well as layers with multiple inputs or outputs. Imagine that, where the Sequential API requires the model to move in a linear fashion through its layers, the Functional API allows much more flexibility. Where Sequential is a straight line, a Functional model is a graph, where the nodes of the layers can connect in many more ways than one. 

In the visual example below, the one possible direction of the movement Sequential model is shown in contrast to a skip connection, which is just one of the many ways a Functional model can be constructed. A skip connection, as you might have guessed, skips some layer in the network and feeds the output to a later layer in the network. Don't worry, you'll be spending more time with skip connections very soon! 

<img src="images/seq_vs_func.png" style="width:350px;height:200px;">

<a name='4-1'></a>
### 4.1 - Load the SIGNS Dataset

As a reminder, the SIGNS dataset is a collection of 6 signs representing numbers from 0 to 5.


```python
# Loading the data (signs)
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_signs_dataset()
```

<img src="images/SIGNS.png" style="width:800px;height:300px;">

The next cell will show you an example of a labelled image in the dataset. Feel free to change the value of `index` below and re-run to see different examples. 


```python
# Example of an image from the dataset
index = 9
plt.imshow(X_train_orig[index])
print ("y = " + str(np.squeeze(Y_train_orig[:, index])))
```

    y = 1



![png](output_29_1.png)


<a name='4-2'></a>
### 4.2 - Split the Data into Train/Test Sets

In Course 2, you built a fully-connected network for this dataset. But since this is an image dataset, it is more natural to apply a ConvNet to it.

To get started, let's examine the shapes of your data. 


```python
X_train = X_train_orig/255.
X_test = X_test_orig/255.
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T
print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
```

    number of training examples = 600
    number of test examples = 150
    X_train shape: (600, 64, 64, 3)
    Y_train shape: (600, 6)
    X_test shape: (150, 64, 64, 3)
    Y_test shape: (150, 6)


<a name='4-3'></a>
### 4.3 - Forward Propagation

In TensorFlow, there are built-in functions that implement the convolution steps for you. By now, you should be familiar with how TensorFlow builds computational graphs. In the [Functional API](https://www.tensorflow.org/guide/keras/functional), you create a graph of layers. This is what allows such great flexibility.

However, the following model could also be defined using the Sequential API since the information flow is on a single line. But don't deviate. What we want you to learn is to use the functional API.

Begin building your graph of layers by creating an input node that functions as a callable object:

- **input_img = tf.keras.Input(shape=input_shape):** 

Then, create a new node in the graph of layers by calling a layer on the `input_img` object: 

- **tf.keras.layers.Conv2D(filters= ... , kernel_size= ... , padding='same')(input_img):** Read the full documentation on [Conv2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D).

- **tf.keras.layers.MaxPool2D(pool_size=(f, f), strides=(s, s), padding='same'):** `MaxPool2D()` downsamples your input using a window of size (f, f) and strides of size (s, s) to carry out max pooling over each window.  For max pooling, you usually operate on a single example at a time and a single channel at a time. Read the full documentation on [MaxPool2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D).

- **tf.keras.layers.ReLU():** computes the elementwise ReLU of Z (which can be any shape). You can read the full documentation on [ReLU](https://www.tensorflow.org/api_docs/python/tf/keras/layers/ReLU).

- **tf.keras.layers.Flatten()**: given a tensor "P", this function takes each training (or test) example in the batch and flattens it into a 1D vector.  

    * If a tensor P has the shape (batch_size,h,w,c), it returns a flattened tensor with shape (batch_size, k), where $k=h \times w \times c$.  "k" equals the product of all the dimension sizes other than the first dimension.
    
    * For example, given a tensor with dimensions [100, 2, 3, 4], it flattens the tensor to be of shape [100, 24], where 24 = 2 * 3 * 4.  You can read the full documentation on [Flatten](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten).

- **tf.keras.layers.Dense(units= ... , activation='softmax')(F):** given the flattened input F, it returns the output computed using a fully connected layer. You can read the full documentation on [Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense).

In the last function above (`tf.keras.layers.Dense()`), the fully connected layer automatically initializes weights in the graph and keeps on training them as you train the model. Hence, you did not need to initialize those weights when initializing the parameters.

Lastly, before creating the model, you'll need to define the output using the last of the function's compositions (in this example, a Dense layer): 

- **outputs = tf.keras.layers.Dense(units=6, activation='softmax')(F)**


#### Window, kernel, filter, pool

The words "kernel" and "filter" are used to refer to the same thing. The word "filter" accounts for the amount of "kernels" that will be used in a single convolution layer. "Pool" is the name of the operation that takes the max or average value of the kernels. 

This is why the parameter `pool_size` refers to `kernel_size`, and you use `(f,f)` to refer to the filter size. 

Pool size and kernel size refer to the same thing in different objects - They refer to the shape of the window where the operation takes place. 

<a name='ex-2'></a>
### Exercise 2 - convolutional_model

Implement the `convolutional_model` function below to build the following model: `CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> DENSE`. Use the functions above! 

Also, plug in the following parameters for all the steps:

 - [Conv2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv2D): Use 8 4 by 4 filters, stride 1, padding is "SAME"
 - [ReLU](https://www.tensorflow.org/api_docs/python/tf/keras/layers/ReLU)
 - [MaxPool2D](https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool2D): Use an 8 by 8 filter size and an 8 by 8 stride, padding is "SAME"
 - **Conv2D**: Use 16 2 by 2 filters, stride 1, padding is "SAME"
 - **ReLU**
 - **MaxPool2D**: Use a 4 by 4 filter size and a 4 by 4 stride, padding is "SAME"
 - [Flatten](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Flatten) the previous output.
 - Fully-connected ([Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense)) layer: Apply a fully connected layer with 6 neurons and a softmax activation. 


```python
# GRADED FUNCTION: convolutional_model

def convolutional_model(input_shape):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> DENSE
    
    Note that for simplicity and grading purposes, you'll hard-code some values
    such as the stride and kernel (filter) sizes. 
    Normally, functions should take these values as function parameters.
    
    Arguments:
    input_img -- input dataset, of shape (input_shape)

    Returns:
    model -- TF Keras model (object containing the information for the entire training process) 
    """

    input_img = tf.keras.Input(shape=input_shape)
    ## CONV2D: 8 filters 4x4, stride of 1, padding 'SAME'
    Z1 = tfl.Conv2D(8,4,padding='SAME')(input_img)
    ## RELU
    A1 = tfl.ReLU()(Z1)
    ## MAXPOOL: window 8x8, stride 8, padding 'SAME'
    P1 = tfl.MaxPool2D(pool_size=(8, 8),strides=8,padding = 'SAME')(A1)
    ## CONV2D: 16 filters 2x2, stride 1, padding 'SAME'
    Z2 = tfl.Conv2D(16,2,padding='SAME')(P1)
    ## RELU
    A2 = tfl.ReLU()(Z2)
    ## MAXPOOL: window 4x4, stride 4, padding 'SAME'
    P2 = tfl.MaxPool2D(pool_size=(4, 4),strides=4,padding = 'SAME')(A2)
    ## FLATTEN
    F = tfl.Flatten()(P2)
    ## Dense layer
    ## 6 neurons in output layer. Hint: one of the arguments should be "activation='softmax'" 
    outputs = tfl.Dense(units=6,activation="softmax")(F)
    # YOUR CODE STARTS HERE
    
    
    # YOUR CODE ENDS HERE
    model = tf.keras.Model(inputs=input_img, outputs=outputs)
    return model
```


```python
conv_model = convolutional_model((64, 64, 3))
conv_model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
conv_model.summary()
    
output = [['InputLayer', [(None, 64, 64, 3)], 0],
        ['Conv2D', (None, 64, 64, 8), 392, 'same', 'linear', 'GlorotUniform'],
        ['ReLU', (None, 64, 64, 8), 0],
        ['MaxPooling2D', (None, 8, 8, 8), 0, (8, 8), (8, 8), 'same'],
        ['Conv2D', (None, 8, 8, 16), 528, 'same', 'linear', 'GlorotUniform'],
        ['ReLU', (None, 8, 8, 16), 0],
        ['MaxPooling2D', (None, 2, 2, 16), 0, (4, 4), (4, 4), 'same'],
        ['Flatten', (None, 64), 0],
        ['Dense', (None, 6), 390, 'softmax']]
    
comparator(summary(conv_model), output)
```

    Model: "functional_2"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_7 (InputLayer)         [(None, 64, 64, 3)]       0         
    _________________________________________________________________
    conv2d_22 (Conv2D)           (None, 64, 64, 8)         392       
    _________________________________________________________________
    re_lu_20 (ReLU)              (None, 64, 64, 8)         0         
    _________________________________________________________________
    max_pooling2d_19 (MaxPooling (None, 8, 8, 8)           0         
    _________________________________________________________________
    conv2d_23 (Conv2D)           (None, 8, 8, 16)          528       
    _________________________________________________________________
    re_lu_21 (ReLU)              (None, 8, 8, 16)          0         
    _________________________________________________________________
    max_pooling2d_20 (MaxPooling (None, 2, 2, 16)          0         
    _________________________________________________________________
    flatten_14 (Flatten)         (None, 64)                0         
    _________________________________________________________________
    dense_15 (Dense)             (None, 6)                 390       
    =================================================================
    Total params: 1,310
    Trainable params: 1,310
    Non-trainable params: 0
    _________________________________________________________________
    [32mAll tests passed![0m


Both the Sequential and Functional APIs return a TF Keras model object. The only difference is how inputs are handled inside the object model! 

<a name='4-4'></a>
### 4.4 - Train the Model


```python
train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train)).batch(64)
test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test)).batch(64)
history = conv_model.fit(train_dataset, epochs=100, validation_data=test_dataset)
```

    Epoch 1/100
    10/10 [==============================] - 1s 110ms/step - loss: 1.4950 - accuracy: 0.5000 - val_loss: 1.2447 - val_accuracy: 0.5600
    Epoch 2/100
    10/10 [==============================] - 1s 101ms/step - loss: 1.1452 - accuracy: 0.5000 - val_loss: 0.9234 - val_accuracy: 0.5600
    Epoch 3/100
    10/10 [==============================] - 1s 100ms/step - loss: 0.8379 - accuracy: 0.5483 - val_loss: 0.7587 - val_accuracy: 0.5733
    Epoch 4/100
    10/10 [==============================] - 1s 110ms/step - loss: 0.7022 - accuracy: 0.6333 - val_loss: 0.6857 - val_accuracy: 0.5867
    Epoch 5/100
    10/10 [==============================] - 1s 100ms/step - loss: 0.6601 - accuracy: 0.6650 - val_loss: 0.6745 - val_accuracy: 0.6400
    Epoch 6/100
    10/10 [==============================] - 1s 100ms/step - loss: 0.6439 - accuracy: 0.7183 - val_loss: 0.6559 - val_accuracy: 0.6800
    Epoch 7/100
    10/10 [==============================] - 1s 100ms/step - loss: 0.6304 - accuracy: 0.7200 - val_loss: 0.6533 - val_accuracy: 0.6733
    Epoch 8/100
    10/10 [==============================] - 1s 110ms/step - loss: 0.6211 - accuracy: 0.7383 - val_loss: 0.6409 - val_accuracy: 0.6867
    Epoch 9/100
    10/10 [==============================] - 1s 100ms/step - loss: 0.6063 - accuracy: 0.7550 - val_loss: 0.6320 - val_accuracy: 0.6933
    Epoch 10/100
    10/10 [==============================] - 1s 109ms/step - loss: 0.5937 - accuracy: 0.7600 - val_loss: 0.6199 - val_accuracy: 0.7200
    Epoch 11/100
    10/10 [==============================] - 1s 101ms/step - loss: 0.5804 - accuracy: 0.7717 - val_loss: 0.6120 - val_accuracy: 0.7067
    Epoch 12/100
    10/10 [==============================] - 1s 100ms/step - loss: 0.5687 - accuracy: 0.7750 - val_loss: 0.6015 - val_accuracy: 0.7133
    Epoch 13/100
    10/10 [==============================] - 1s 110ms/step - loss: 0.5561 - accuracy: 0.7833 - val_loss: 0.5903 - val_accuracy: 0.7200
    Epoch 14/100
    10/10 [==============================] - 1s 100ms/step - loss: 0.5433 - accuracy: 0.7933 - val_loss: 0.5802 - val_accuracy: 0.7267
    Epoch 15/100
    10/10 [==============================] - 1s 101ms/step - loss: 0.5309 - accuracy: 0.8000 - val_loss: 0.5714 - val_accuracy: 0.7333
    Epoch 16/100
    10/10 [==============================] - 1s 110ms/step - loss: 0.5187 - accuracy: 0.8050 - val_loss: 0.5633 - val_accuracy: 0.7467
    Epoch 17/100
    10/10 [==============================] - 1s 110ms/step - loss: 0.5064 - accuracy: 0.8133 - val_loss: 0.5545 - val_accuracy: 0.7533
    Epoch 18/100
    10/10 [==============================] - 1s 110ms/step - loss: 0.4945 - accuracy: 0.8250 - val_loss: 0.5482 - val_accuracy: 0.7467
    Epoch 19/100
    10/10 [==============================] - 1s 110ms/step - loss: 0.4836 - accuracy: 0.8250 - val_loss: 0.5409 - val_accuracy: 0.7400
    Epoch 20/100
    10/10 [==============================] - 1s 110ms/step - loss: 0.4727 - accuracy: 0.8283 - val_loss: 0.5353 - val_accuracy: 0.7400
    Epoch 21/100
    10/10 [==============================] - 1s 110ms/step - loss: 0.4619 - accuracy: 0.8317 - val_loss: 0.5284 - val_accuracy: 0.7467
    Epoch 22/100
    10/10 [==============================] - 1s 110ms/step - loss: 0.4515 - accuracy: 0.8367 - val_loss: 0.5209 - val_accuracy: 0.7400
    Epoch 23/100
    10/10 [==============================] - 1s 110ms/step - loss: 0.4408 - accuracy: 0.8383 - val_loss: 0.5125 - val_accuracy: 0.7267
    Epoch 24/100
    10/10 [==============================] - 1s 110ms/step - loss: 0.4309 - accuracy: 0.8433 - val_loss: 0.5054 - val_accuracy: 0.7333
    Epoch 25/100
    10/10 [==============================] - 1s 100ms/step - loss: 0.4216 - accuracy: 0.8467 - val_loss: 0.4988 - val_accuracy: 0.7333
    Epoch 26/100
    10/10 [==============================] - 1s 100ms/step - loss: 0.4127 - accuracy: 0.8467 - val_loss: 0.4926 - val_accuracy: 0.7333
    Epoch 27/100
    10/10 [==============================] - 1s 110ms/step - loss: 0.4040 - accuracy: 0.8517 - val_loss: 0.4858 - val_accuracy: 0.7267
    Epoch 28/100
    10/10 [==============================] - 1s 110ms/step - loss: 0.3954 - accuracy: 0.8550 - val_loss: 0.4787 - val_accuracy: 0.7467
    Epoch 29/100
    10/10 [==============================] - 1s 101ms/step - loss: 0.3869 - accuracy: 0.8617 - val_loss: 0.4721 - val_accuracy: 0.7600
    Epoch 30/100
    10/10 [==============================] - 1s 110ms/step - loss: 0.3790 - accuracy: 0.8667 - val_loss: 0.4659 - val_accuracy: 0.7600
    Epoch 31/100
    10/10 [==============================] - 1s 101ms/step - loss: 0.3713 - accuracy: 0.8683 - val_loss: 0.4606 - val_accuracy: 0.7600
    Epoch 32/100
    10/10 [==============================] - 1s 100ms/step - loss: 0.3643 - accuracy: 0.8700 - val_loss: 0.4548 - val_accuracy: 0.7600
    Epoch 33/100
    10/10 [==============================] - 1s 110ms/step - loss: 0.3573 - accuracy: 0.8683 - val_loss: 0.4494 - val_accuracy: 0.7533
    Epoch 34/100
    10/10 [==============================] - 1s 102ms/step - loss: 0.3507 - accuracy: 0.8683 - val_loss: 0.4435 - val_accuracy: 0.7600
    Epoch 35/100
    10/10 [==============================] - 1s 101ms/step - loss: 0.3445 - accuracy: 0.8717 - val_loss: 0.4364 - val_accuracy: 0.7733
    Epoch 36/100
    10/10 [==============================] - 1s 100ms/step - loss: 0.3368 - accuracy: 0.8733 - val_loss: 0.4236 - val_accuracy: 0.7733
    Epoch 37/100
    10/10 [==============================] - 1s 110ms/step - loss: 0.3296 - accuracy: 0.8783 - val_loss: 0.4176 - val_accuracy: 0.7800
    Epoch 38/100
    10/10 [==============================] - 1s 100ms/step - loss: 0.3235 - accuracy: 0.8817 - val_loss: 0.4134 - val_accuracy: 0.7800
    Epoch 39/100
    10/10 [==============================] - 1s 110ms/step - loss: 0.3168 - accuracy: 0.8833 - val_loss: 0.4097 - val_accuracy: 0.7733
    Epoch 40/100
    10/10 [==============================] - 1s 110ms/step - loss: 0.3107 - accuracy: 0.8867 - val_loss: 0.4038 - val_accuracy: 0.7867
    Epoch 41/100
    10/10 [==============================] - 1s 110ms/step - loss: 0.3049 - accuracy: 0.8900 - val_loss: 0.3981 - val_accuracy: 0.7933
    Epoch 42/100
    10/10 [==============================] - 1s 101ms/step - loss: 0.2993 - accuracy: 0.8900 - val_loss: 0.3917 - val_accuracy: 0.7933
    Epoch 43/100
    10/10 [==============================] - 1s 101ms/step - loss: 0.2940 - accuracy: 0.8883 - val_loss: 0.3864 - val_accuracy: 0.7933
    Epoch 44/100
    10/10 [==============================] - 1s 110ms/step - loss: 0.2890 - accuracy: 0.8967 - val_loss: 0.3813 - val_accuracy: 0.8000
    Epoch 45/100
    10/10 [==============================] - 1s 109ms/step - loss: 0.2844 - accuracy: 0.8983 - val_loss: 0.3769 - val_accuracy: 0.8067
    Epoch 46/100
    10/10 [==============================] - 1s 110ms/step - loss: 0.2800 - accuracy: 0.9000 - val_loss: 0.3728 - val_accuracy: 0.8200
    Epoch 47/100
    10/10 [==============================] - 1s 110ms/step - loss: 0.2757 - accuracy: 0.9017 - val_loss: 0.3686 - val_accuracy: 0.8200
    Epoch 48/100
    10/10 [==============================] - 1s 110ms/step - loss: 0.2715 - accuracy: 0.9017 - val_loss: 0.3644 - val_accuracy: 0.8200
    Epoch 49/100
    10/10 [==============================] - 1s 110ms/step - loss: 0.2673 - accuracy: 0.9033 - val_loss: 0.3601 - val_accuracy: 0.8200
    Epoch 50/100
    10/10 [==============================] - 1s 109ms/step - loss: 0.2632 - accuracy: 0.9050 - val_loss: 0.3552 - val_accuracy: 0.8200
    Epoch 51/100
    10/10 [==============================] - 1s 100ms/step - loss: 0.2592 - accuracy: 0.9067 - val_loss: 0.3506 - val_accuracy: 0.8200
    Epoch 52/100
    10/10 [==============================] - 1s 109ms/step - loss: 0.2552 - accuracy: 0.9100 - val_loss: 0.3460 - val_accuracy: 0.8200
    Epoch 53/100
    10/10 [==============================] - 1s 101ms/step - loss: 0.2514 - accuracy: 0.9117 - val_loss: 0.3418 - val_accuracy: 0.8200
    Epoch 54/100
    10/10 [==============================] - 1s 101ms/step - loss: 0.2479 - accuracy: 0.9133 - val_loss: 0.3378 - val_accuracy: 0.8200
    Epoch 55/100
    10/10 [==============================] - 1s 110ms/step - loss: 0.2441 - accuracy: 0.9133 - val_loss: 0.3336 - val_accuracy: 0.8267
    Epoch 56/100
    10/10 [==============================] - 1s 101ms/step - loss: 0.2407 - accuracy: 0.9183 - val_loss: 0.3303 - val_accuracy: 0.8267
    Epoch 57/100
    10/10 [==============================] - 1s 101ms/step - loss: 0.2373 - accuracy: 0.9217 - val_loss: 0.3267 - val_accuracy: 0.8267
    Epoch 58/100
    10/10 [==============================] - 1s 101ms/step - loss: 0.2339 - accuracy: 0.9217 - val_loss: 0.3230 - val_accuracy: 0.8267
    Epoch 59/100
    10/10 [==============================] - 1s 101ms/step - loss: 0.2306 - accuracy: 0.9200 - val_loss: 0.3191 - val_accuracy: 0.8333
    Epoch 60/100
    10/10 [==============================] - 1s 101ms/step - loss: 0.2278 - accuracy: 0.9233 - val_loss: 0.3160 - val_accuracy: 0.8333
    Epoch 61/100
    10/10 [==============================] - 1s 101ms/step - loss: 0.2248 - accuracy: 0.9283 - val_loss: 0.3128 - val_accuracy: 0.8400
    Epoch 62/100
    10/10 [==============================] - 1s 100ms/step - loss: 0.2219 - accuracy: 0.9283 - val_loss: 0.3095 - val_accuracy: 0.8533
    Epoch 63/100
    10/10 [==============================] - 1s 100ms/step - loss: 0.2190 - accuracy: 0.9283 - val_loss: 0.3060 - val_accuracy: 0.8533
    Epoch 64/100
    10/10 [==============================] - 1s 100ms/step - loss: 0.2163 - accuracy: 0.9317 - val_loss: 0.3025 - val_accuracy: 0.8533
    Epoch 65/100
    10/10 [==============================] - 1s 109ms/step - loss: 0.2138 - accuracy: 0.9317 - val_loss: 0.2996 - val_accuracy: 0.8600
    Epoch 66/100
    10/10 [==============================] - 1s 101ms/step - loss: 0.2114 - accuracy: 0.9350 - val_loss: 0.2968 - val_accuracy: 0.8600
    Epoch 67/100
    10/10 [==============================] - 1s 110ms/step - loss: 0.2090 - accuracy: 0.9367 - val_loss: 0.2940 - val_accuracy: 0.8600
    Epoch 68/100
    10/10 [==============================] - 1s 109ms/step - loss: 0.2066 - accuracy: 0.9367 - val_loss: 0.2911 - val_accuracy: 0.8667
    Epoch 69/100
    10/10 [==============================] - 1s 101ms/step - loss: 0.2043 - accuracy: 0.9367 - val_loss: 0.2891 - val_accuracy: 0.8667
    Epoch 70/100
    10/10 [==============================] - 1s 101ms/step - loss: 0.2021 - accuracy: 0.9367 - val_loss: 0.2867 - val_accuracy: 0.8667
    Epoch 71/100
    10/10 [==============================] - 1s 109ms/step - loss: 0.1999 - accuracy: 0.9383 - val_loss: 0.2842 - val_accuracy: 0.8667
    Epoch 72/100
    10/10 [==============================] - 1s 101ms/step - loss: 0.1977 - accuracy: 0.9383 - val_loss: 0.2817 - val_accuracy: 0.8667
    Epoch 73/100
    10/10 [==============================] - 1s 101ms/step - loss: 0.1955 - accuracy: 0.9383 - val_loss: 0.2790 - val_accuracy: 0.8733
    Epoch 74/100
    10/10 [==============================] - 1s 110ms/step - loss: 0.1933 - accuracy: 0.9400 - val_loss: 0.2769 - val_accuracy: 0.8733
    Epoch 75/100
    10/10 [==============================] - 1s 100ms/step - loss: 0.1913 - accuracy: 0.9400 - val_loss: 0.2744 - val_accuracy: 0.8733
    Epoch 76/100
    10/10 [==============================] - 1s 100ms/step - loss: 0.1892 - accuracy: 0.9400 - val_loss: 0.2721 - val_accuracy: 0.8800
    Epoch 77/100
    10/10 [==============================] - 1s 100ms/step - loss: 0.1873 - accuracy: 0.9400 - val_loss: 0.2700 - val_accuracy: 0.8800
    Epoch 78/100
    10/10 [==============================] - 1s 109ms/step - loss: 0.1852 - accuracy: 0.9433 - val_loss: 0.2675 - val_accuracy: 0.8867
    Epoch 79/100
    10/10 [==============================] - 1s 100ms/step - loss: 0.1832 - accuracy: 0.9433 - val_loss: 0.2650 - val_accuracy: 0.8867
    Epoch 80/100
    10/10 [==============================] - 1s 110ms/step - loss: 0.1813 - accuracy: 0.9450 - val_loss: 0.2630 - val_accuracy: 0.8867
    Epoch 81/100
    10/10 [==============================] - 1s 109ms/step - loss: 0.1795 - accuracy: 0.9450 - val_loss: 0.2612 - val_accuracy: 0.8867
    Epoch 82/100
    10/10 [==============================] - 1s 101ms/step - loss: 0.1776 - accuracy: 0.9433 - val_loss: 0.2593 - val_accuracy: 0.8867
    Epoch 83/100
    10/10 [==============================] - 1s 101ms/step - loss: 0.1757 - accuracy: 0.9433 - val_loss: 0.2573 - val_accuracy: 0.8867
    Epoch 84/100
    10/10 [==============================] - 1s 101ms/step - loss: 0.1739 - accuracy: 0.9450 - val_loss: 0.2551 - val_accuracy: 0.8867
    Epoch 85/100
    10/10 [==============================] - 1s 109ms/step - loss: 0.1722 - accuracy: 0.9450 - val_loss: 0.2536 - val_accuracy: 0.8867
    Epoch 86/100
    10/10 [==============================] - 1s 110ms/step - loss: 0.1706 - accuracy: 0.9467 - val_loss: 0.2518 - val_accuracy: 0.8867
    Epoch 87/100
    10/10 [==============================] - 1s 109ms/step - loss: 0.1689 - accuracy: 0.9467 - val_loss: 0.2498 - val_accuracy: 0.9000
    Epoch 88/100
    10/10 [==============================] - 1s 101ms/step - loss: 0.1673 - accuracy: 0.9467 - val_loss: 0.2485 - val_accuracy: 0.9067
    Epoch 89/100
    10/10 [==============================] - 1s 100ms/step - loss: 0.1658 - accuracy: 0.9467 - val_loss: 0.2473 - val_accuracy: 0.9067
    Epoch 90/100
    10/10 [==============================] - 1s 100ms/step - loss: 0.1642 - accuracy: 0.9483 - val_loss: 0.2456 - val_accuracy: 0.9067
    Epoch 91/100
    10/10 [==============================] - 1s 101ms/step - loss: 0.1627 - accuracy: 0.9467 - val_loss: 0.2440 - val_accuracy: 0.9067
    Epoch 92/100
    10/10 [==============================] - 1s 109ms/step - loss: 0.1612 - accuracy: 0.9483 - val_loss: 0.2425 - val_accuracy: 0.9067
    Epoch 93/100
    10/10 [==============================] - 1s 101ms/step - loss: 0.1598 - accuracy: 0.9500 - val_loss: 0.2410 - val_accuracy: 0.9067
    Epoch 94/100
    10/10 [==============================] - 1s 101ms/step - loss: 0.1585 - accuracy: 0.9500 - val_loss: 0.2406 - val_accuracy: 0.9067
    Epoch 95/100
    10/10 [==============================] - 1s 100ms/step - loss: 0.1570 - accuracy: 0.9500 - val_loss: 0.2388 - val_accuracy: 0.9133
    Epoch 96/100
    10/10 [==============================] - 1s 110ms/step - loss: 0.1555 - accuracy: 0.9517 - val_loss: 0.2376 - val_accuracy: 0.9067
    Epoch 97/100
    10/10 [==============================] - 1s 101ms/step - loss: 0.1542 - accuracy: 0.9517 - val_loss: 0.2364 - val_accuracy: 0.9133
    Epoch 98/100
    10/10 [==============================] - 1s 101ms/step - loss: 0.1529 - accuracy: 0.9517 - val_loss: 0.2352 - val_accuracy: 0.9133
    Epoch 99/100
    10/10 [==============================] - 1s 110ms/step - loss: 0.1516 - accuracy: 0.9533 - val_loss: 0.2338 - val_accuracy: 0.9067
    Epoch 100/100
    10/10 [==============================] - 1s 109ms/step - loss: 0.1505 - accuracy: 0.9533 - val_loss: 0.2335 - val_accuracy: 0.9133


<a name='5'></a>
## 5 - History Object 

The history object is an output of the `.fit()` operation, and provides a record of all the loss and metric values in memory. It's stored as a dictionary that you can retrieve at `history.history`: 


```python
history.history
```




    {'loss': [1.495040774345398,
      1.145181655883789,
      0.8378946781158447,
      0.7022131085395813,
      0.6601147055625916,
      0.6438847780227661,
      0.6304004192352295,
      0.6210683584213257,
      0.6062697172164917,
      0.5937250852584839,
      0.5804409980773926,
      0.5686632394790649,
      0.5560791492462158,
      0.5432971715927124,
      0.5308559536933899,
      0.5187262296676636,
      0.506360650062561,
      0.4945167601108551,
      0.48358389735221863,
      0.4727034568786621,
      0.46194934844970703,
      0.4514774978160858,
      0.4408019483089447,
      0.4308643639087677,
      0.42164671421051025,
      0.41272228956222534,
      0.4040181338787079,
      0.39535045623779297,
      0.38692039251327515,
      0.378950297832489,
      0.3713197112083435,
      0.36431121826171875,
      0.35734379291534424,
      0.35074105858802795,
      0.3444783389568329,
      0.33675727248191833,
      0.32957738637924194,
      0.3234633505344391,
      0.316845178604126,
      0.31074070930480957,
      0.30489861965179443,
      0.2993386685848236,
      0.29395559430122375,
      0.2890087366104126,
      0.2844134271144867,
      0.28001898527145386,
      0.27569931745529175,
      0.271512508392334,
      0.2673491835594177,
      0.26324760913848877,
      0.2591713070869446,
      0.2552381157875061,
      0.2514123320579529,
      0.2478540986776352,
      0.24413976073265076,
      0.24067431688308716,
      0.2372746467590332,
      0.23393231630325317,
      0.23061133921146393,
      0.22775942087173462,
      0.22478218376636505,
      0.2218765765428543,
      0.2189977765083313,
      0.21626083552837372,
      0.21382886171340942,
      0.2114141583442688,
      0.2089749574661255,
      0.20662224292755127,
      0.2043154090642929,
      0.2020684778690338,
      0.19987741112709045,
      0.1977032572031021,
      0.1954997479915619,
      0.1933361440896988,
      0.19131669402122498,
      0.18923339247703552,
      0.18726977705955505,
      0.18521425127983093,
      0.18318656086921692,
      0.18127410113811493,
      0.17945826053619385,
      0.177595853805542,
      0.17567716538906097,
      0.17385047674179077,
      0.17220069468021393,
      0.1705591082572937,
      0.16887769103050232,
      0.16729632019996643,
      0.16584426164627075,
      0.16415110230445862,
      0.1626741886138916,
      0.16118071973323822,
      0.15976759791374207,
      0.15850912034511566,
      0.15703020989894867,
      0.15548725426197052,
      0.15417933464050293,
      0.15294264256954193,
      0.15163347125053406,
      0.1505042463541031],
     'accuracy': [0.5,
      0.5,
      0.5483333468437195,
      0.6333333253860474,
      0.6650000214576721,
      0.7183333039283752,
      0.7200000286102295,
      0.7383333444595337,
      0.7549999952316284,
      0.7599999904632568,
      0.7716666460037231,
      0.7749999761581421,
      0.7833333611488342,
      0.7933333516120911,
      0.800000011920929,
      0.8050000071525574,
      0.8133333325386047,
      0.824999988079071,
      0.824999988079071,
      0.82833331823349,
      0.8316666483879089,
      0.8366666436195374,
      0.8383333086967468,
      0.8433333039283752,
      0.846666693687439,
      0.846666693687439,
      0.8516666889190674,
      0.8550000190734863,
      0.8616666793823242,
      0.8666666746139526,
      0.8683333396911621,
      0.8700000047683716,
      0.8683333396911621,
      0.8683333396911621,
      0.871666669845581,
      0.8733333349227905,
      0.878333330154419,
      0.8816666603088379,
      0.8833333253860474,
      0.8866666555404663,
      0.8899999856948853,
      0.8899999856948853,
      0.8883333206176758,
      0.8966666460037231,
      0.8983333110809326,
      0.8999999761581421,
      0.9016666412353516,
      0.9016666412353516,
      0.903333306312561,
      0.9049999713897705,
      0.9066666960716248,
      0.9100000262260437,
      0.9116666913032532,
      0.9133333563804626,
      0.9133333563804626,
      0.9183333516120911,
      0.92166668176651,
      0.92166668176651,
      0.9200000166893005,
      0.9233333468437195,
      0.9283333420753479,
      0.9283333420753479,
      0.9283333420753479,
      0.9316666722297668,
      0.9316666722297668,
      0.9350000023841858,
      0.9366666674613953,
      0.9366666674613953,
      0.9366666674613953,
      0.9366666674613953,
      0.9383333325386047,
      0.9383333325386047,
      0.9383333325386047,
      0.9399999976158142,
      0.9399999976158142,
      0.9399999976158142,
      0.9399999976158142,
      0.9433333277702332,
      0.9433333277702332,
      0.9449999928474426,
      0.9449999928474426,
      0.9433333277702332,
      0.9433333277702332,
      0.9449999928474426,
      0.9449999928474426,
      0.9466666579246521,
      0.9466666579246521,
      0.9466666579246521,
      0.9466666579246521,
      0.9483333230018616,
      0.9466666579246521,
      0.9483333230018616,
      0.949999988079071,
      0.949999988079071,
      0.949999988079071,
      0.9516666531562805,
      0.9516666531562805,
      0.9516666531562805,
      0.95333331823349,
      0.95333331823349],
     'val_loss': [1.2446507215499878,
      0.9233507513999939,
      0.7586942911148071,
      0.6856560111045837,
      0.6744608283042908,
      0.6559008955955505,
      0.6532639861106873,
      0.6408910751342773,
      0.6319581866264343,
      0.6199132204055786,
      0.6120439171791077,
      0.601509153842926,
      0.5903422832489014,
      0.5802014470100403,
      0.5714151263237,
      0.5632825493812561,
      0.5545033812522888,
      0.548190712928772,
      0.5409426093101501,
      0.5352929830551147,
      0.5284141302108765,
      0.5208685398101807,
      0.5124621391296387,
      0.5053731203079224,
      0.498831570148468,
      0.49261993169784546,
      0.4857944846153259,
      0.47871285676956177,
      0.47206175327301025,
      0.46588021516799927,
      0.46061456203460693,
      0.4547812342643738,
      0.44936293363571167,
      0.4435111880302429,
      0.43644821643829346,
      0.4236411452293396,
      0.4175731837749481,
      0.41342034935951233,
      0.40974152088165283,
      0.4037821888923645,
      0.3980557322502136,
      0.39167487621307373,
      0.38635528087615967,
      0.38128983974456787,
      0.37693873047828674,
      0.3728133738040924,
      0.36857134103775024,
      0.3643718659877777,
      0.36010119318962097,
      0.3552374839782715,
      0.35059434175491333,
      0.3460446298122406,
      0.34177258610725403,
      0.3377738893032074,
      0.3335697054862976,
      0.3302861452102661,
      0.3266657888889313,
      0.3229627311229706,
      0.3190670907497406,
      0.31599587202072144,
      0.31275346875190735,
      0.30946046113967896,
      0.3059648275375366,
      0.30247417092323303,
      0.29957494139671326,
      0.2967551052570343,
      0.29395803809165955,
      0.2910933792591095,
      0.2890804409980774,
      0.2867203950881958,
      0.2841845452785492,
      0.28167209029197693,
      0.279020756483078,
      0.2768748998641968,
      0.27443262934684753,
      0.27208033204078674,
      0.2700115144252777,
      0.2675478160381317,
      0.26503103971481323,
      0.26299503445625305,
      0.2611587643623352,
      0.25930899381637573,
      0.2572915554046631,
      0.2551434636116028,
      0.2536245584487915,
      0.25175830721855164,
      0.24983017146587372,
      0.248478502035141,
      0.24731183052062988,
      0.24557220935821533,
      0.24400804936885834,
      0.24250304698944092,
      0.24104218184947968,
      0.24057841300964355,
      0.23875124752521515,
      0.23755833506584167,
      0.23635436594486237,
      0.23521967232227325,
      0.23375198245048523,
      0.2334655225276947],
     'val_accuracy': [0.5600000023841858,
      0.5600000023841858,
      0.5733333230018616,
      0.5866666436195374,
      0.6399999856948853,
      0.6800000071525574,
      0.6733333468437195,
      0.6866666674613953,
      0.6933333277702332,
      0.7200000286102295,
      0.7066666483879089,
      0.7133333086967468,
      0.7200000286102295,
      0.7266666889190674,
      0.7333333492279053,
      0.746666669845581,
      0.753333330154419,
      0.746666669845581,
      0.7400000095367432,
      0.7400000095367432,
      0.746666669845581,
      0.7400000095367432,
      0.7266666889190674,
      0.7333333492279053,
      0.7333333492279053,
      0.7333333492279053,
      0.7266666889190674,
      0.746666669845581,
      0.7599999904632568,
      0.7599999904632568,
      0.7599999904632568,
      0.7599999904632568,
      0.753333330154419,
      0.7599999904632568,
      0.7733333110809326,
      0.7733333110809326,
      0.7799999713897705,
      0.7799999713897705,
      0.7733333110809326,
      0.7866666913032532,
      0.7933333516120911,
      0.7933333516120911,
      0.7933333516120911,
      0.800000011920929,
      0.8066666722297668,
      0.8199999928474426,
      0.8199999928474426,
      0.8199999928474426,
      0.8199999928474426,
      0.8199999928474426,
      0.8199999928474426,
      0.8199999928474426,
      0.8199999928474426,
      0.8199999928474426,
      0.8266666531562805,
      0.8266666531562805,
      0.8266666531562805,
      0.8266666531562805,
      0.8333333134651184,
      0.8333333134651184,
      0.8399999737739563,
      0.8533333539962769,
      0.8533333539962769,
      0.8533333539962769,
      0.8600000143051147,
      0.8600000143051147,
      0.8600000143051147,
      0.8666666746139526,
      0.8666666746139526,
      0.8666666746139526,
      0.8666666746139526,
      0.8666666746139526,
      0.8733333349227905,
      0.8733333349227905,
      0.8733333349227905,
      0.8799999952316284,
      0.8799999952316284,
      0.8866666555404663,
      0.8866666555404663,
      0.8866666555404663,
      0.8866666555404663,
      0.8866666555404663,
      0.8866666555404663,
      0.8866666555404663,
      0.8866666555404663,
      0.8866666555404663,
      0.8999999761581421,
      0.9066666960716248,
      0.9066666960716248,
      0.9066666960716248,
      0.9066666960716248,
      0.9066666960716248,
      0.9066666960716248,
      0.9066666960716248,
      0.9133333563804626,
      0.9066666960716248,
      0.9133333563804626,
      0.9133333563804626,
      0.9066666960716248,
      0.9133333563804626]}



Now visualize the loss over time using `history.history`: 


```python
# The history.history["loss"] entry is a dictionary with as many values as epochs that the
# model was trained on. 
df_loss_acc = pd.DataFrame(history.history)
df_loss= df_loss_acc[['loss','val_loss']]
df_loss.rename(columns={'loss':'train','val_loss':'validation'},inplace=True)
df_acc= df_loss_acc[['accuracy','val_accuracy']]
df_acc.rename(columns={'accuracy':'train','val_accuracy':'validation'},inplace=True)
df_loss.plot(title='Model loss',figsize=(12,8)).set(xlabel='Epoch',ylabel='Loss')
df_acc.plot(title='Model Accuracy',figsize=(12,8)).set(xlabel='Epoch',ylabel='Accuracy')
```




    [Text(0, 0.5, 'Accuracy'), Text(0.5, 0, 'Epoch')]




![png](output_42_1.png)



![png](output_42_2.png)


**Congratulations**! You've finished the assignment and built two models: One that recognizes  smiles, and another that recognizes SIGN language with almost 80% accuracy on the test set. In addition to that, you now also understand the applications of two Keras APIs: Sequential and Functional. Nicely done! 

By now, you know a bit about how the Functional API works and may have glimpsed the possibilities. In your next assignment, you'll really get a feel for its power when you get the opportunity to build a very deep ConvNet, using ResNets! 

<a name='6'></a>
## 6 - Bibliography

You're always encouraged to read the official documentation. To that end, you can find the docs for the Sequential and Functional APIs here: 

https://www.tensorflow.org/guide/keras/sequential_model

https://www.tensorflow.org/guide/keras/functional


```python

```


```python

```
