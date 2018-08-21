# -*- coding: utf-8 -*-
"""
Created on Tue Aug 14 11:45:04 2018

@author: Matthew
"""

# Import statements
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.contrib.keras.api.keras as keras
from sklearn.model_selection import train_test_split


# Load MNIST dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Normalize pixels between 0 - 1
X_train = X_train / 255
X_test = X_test / 255



def display_mnist(x, y, index):
    
    bw = plt.get_cmap('gray')
    
    plt.imshow(x[index], cmap=bw)
    plt.title(str(y[index]))
    plt.show()


# Use to look at a specific image within the X_train MNIST dataset
#display_mnist(X_train, y_train, index = 7)


def flatten_images(train, test):
    
    # Flatten training samples
    num_train_samples = train.shape[0]
    im_len = train.shape[1]
    im_width = train.shape[2]
    
    # Flatten images from (28 x 28) to (784 x 1)
    train = train.reshape((num_train_samples, im_len * im_width))
    
    # Flatten testing samples
    num_test_samples = test.shape[0]
    test = test.reshape((num_test_samples, im_len * im_width))
    
    # Return flattened images
    return(train, test)




def baseline_model(input_shape, num_classes):
    
	 # create model
    model = keras.models.Sequential()
    
    # Input dense layer 
    model.add(keras.layers.Dense(units = 256,
                                 input_dim = input_shape,
                                 activation='relu'))
    
    # Output dense layer
    model.add(keras.layers.Dense(units = num_classes, 
                                 activation='softmax'))
	
    # Compile model
    model.compile(loss = 'categorical_crossentropy',
                  optimizer = 'adam',
                  metrics = ['accuracy'])
    
    return(model)
   
    

def test_baseline_model(X_train, X_test, y_train, y_test):
    
    # Prepare data format
    X_train, X_test = flatten_images(X_train, X_test)
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)
    
    # Create validation data
    X_trn, X_val, y_trn, y_val = train_test_split(X_train, y_train,
                                                  stratify = y_train,
                                                  test_size = 0.2)
        
    # Create and compile model
    input_shape = X_trn.shape[1]        # 784
    num_classes = y_trn.shape[1]        # 10
    
    # Create and compile baseline neural network
    model = baseline_model(input_shape, num_classes)
    
    # Fit the model
    model.fit(X_trn, y_trn,
              validation_data=(X_val, y_val),
              epochs = 10,
              batch_size = 128)
    
    
    # Evaluation of the model
    scores = model.evaluate(X_test, y_test)
    print("Accuracy: {}%".format(scores[1] * 100))
    
    return(model)



def cnn_model(input_shape, num_classes):
    
    # create model
    model = keras.models.Sequential()
    
    # Convolution input layer
    model.add(keras.layers.Conv2D(input_shape = input_shape,
                                  filters = 6,
                                  kernel_size = (5, 5),
                                  activation = 'relu'))
    
    # Pooling layer
    model.add(keras.layers.MaxPooling2D(pool_size = (2, 2)))
    
    # Dropout
    model.add(keras.layers.Dropout(rate = 0.1))
    
    # Convolution input layer
    model.add(keras.layers.Conv2D(filters = 12,
                                  kernel_size = (3, 3),
                                  activation = 'relu'))

    # Pooling layer
    model.add(keras.layers.MaxPooling2D(pool_size = (2, 2)))
    
    # Dropout
    model.add(keras.layers.Dropout(rate = 0.1))
    
    # Flatten for input to dense layers
    model.add(keras.layers.Flatten())
    
    # Input dense layer 
    model.add(keras.layers.Dense(units = 256,
                                 activation='relu'))
    
    # Output dense layer
    model.add(keras.layers.Dense(units = num_classes, 
                                 activation='softmax'))
	
    # Compile model
    model.compile(loss = 'categorical_crossentropy',
                  optimizer = 'adam',
                  metrics = ['accuracy'])
    
    return(model)



def test_cnn_model(X_train, X_test, y_train, y_test):
    
    # Prepare data format
    X_train = np.expand_dims(X_train, -1)
    X_test = np.expand_dims(X_test, -1)
    y_train = keras.utils.to_categorical(y_train)
    y_test = keras.utils.to_categorical(y_test)
    
    # Create validation data
    X_trn, X_val, y_trn, y_val = train_test_split(X_train, y_train,
                                                  stratify = y_train,
                                                  test_size = 0.2)
        
    # Explicity state input shape
    length = X_trn.shape[1]
    width = X_trn.shape[2]
    channels = X_trn.shape[3]
    
    # Create and compile model
    input_shape = (length, width, channels)
    num_classes = y_trn.shape[1]
    
    # Get model architecture
    model = cnn_model(input_shape, num_classes)
    
    # Fit the model
    model.fit(X_trn, y_trn,
              validation_data=(X_val, y_val),
              epochs = 10,
              batch_size = 128)
    
    # Evaluation of the model
    scores = model.evaluate(X_test, y_test)
    print("Accuracy: {}%".format(scores[1] * 100))
    
    return(model)



def find_errors(model, X_test, y_test): 
    
    # Convert to CNN format
    X_test = np.expand_dims(X_test, -1)
    
    # Indicies which are incorrect
    idxs = []
    error_predictions = []
    
    # Perform predictions on test data
    vals = model.predict(X_test)
    
    # For each test image we predicted
    for i in range(len(vals)):
        preds = vals[i]        
        
        # Convert one-hot encoding back to a single number
        class_pred = np.argmax(preds)
        class_truth = y_test[i]
        
        # Compare our prediction to the known true label
        if class_pred != class_truth:
            
            # If they're different - this was one of the incorrect images
            idxs.append(i)
            error_predictions.append(class_pred)
    
    # Strip         
    X_errors = X_test[idxs]
    
    # Inverse of 'expand_dims' reduces the dimentionality, so goes from
    # samples x 28 x 28 x 1    ---->     samples x 28 x 28
    X_errors = X_errors.squeeze(axis = -1)
    
    # Labels for the incorrect images
    y_errors = y_test[idxs]
    
    # Return incorrect images, their true label, and the prediction
    return(X_errors, y_errors, error_predictions)
   

def visualize_error(x, y, preds, index):
    
    print('Image Label: {}'.format(y[index]))
    print('Neural Network Prediction: {}'.format(preds[index]))
    plt.imshow(x[index], cmap = plt.get_cmap('gray'))
    plt.show()
    


#model = test_baseline_model(X_train, X_test, y_train, y_test)
#model = test_cnn_model(X_train, X_test, y_train, y_test)
X_errors, y_errors, error_preds = find_errors(model, X_test, y_test)
visualize_error(X_errors, y_errors, error_preds, index = 7)
    