# Tech preamble:

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Conv1D, LSTM, concatenate, MaxPooling1D, Dropout, Flatten, Activation, BatchNormalization, LeakyReLU
)
from tensorflow.keras import Sequential
from tensorflow.keras.optimizers import SGD, Adam
import tensorflow.keras.initializers as tfi
import tensorflow.keras.regularizers as tfr


def set_CNN_fc(
    input_length, 
    n_features, 
    max_pooling_threshold,
    CNN_filters, 
    CNN_kernel_sizes, 
    batch_normalization,
    alpha,
    fc_units, 
    fc_activation,
    output_activation,
    CNN_weight_init,
    CNN_bias_init,
    fc_weight_init,
    fc_bias_init,
    CNN_weight_reg,
    CNN_bias_reg,
    fc_weight_reg,
    fc_bias_reg,
    learning_rate, 
    loss_function
):
    """
    Sets up CNN/fc model. Can be used for multi-run experiments.
    
    Parameters:
    ===========
    input_length: int
        Specify the number of time steps as input length.
    n_features: int
        Number of input features.
    max_pooling_threshold: int
        Specify threshold for input length: If input length exceeds threshold, apply max pooling with poolsize=2.
    CNN_filters: array of int
        Specify the number of feature maps in all CNN layers.
    CNN_kernel_sizes: array of int
        Specify the filter sizes in all CNN layers.
    batch_normalization: boolean
        Choose, whether or not to add batch normalization directly after each convolution operation.
    alpha: float
        Specify leak rate for leaky ReLU activation in convolution blocks.
    fc_units: array of int
        Specify the number of units in all hidden fc layers.
    fc_activation: Str (e.g. 'sigmoid')
        Specify activation function in hidden fc layers.
    output_activation: Str (e.g. 'linear')
        Specify activation function for output unit.
    CNN_weight_init: Str (e.g. 'glorot_uniform' as default)
        Specify how to initialize kernel weights in CNN layers.
    CNN_bias_init: Str (e.g. 'zeros' as default)
        Specify how to initialize biases in CNN layers.
    fc_weight_init: Str (e.g. 'glorot_uniform' as default)
        Specify how to initialize weights in fc layers.
    fc_bias_init: Str (e.g. 'zeros' as default)
        Specify how to initialize biases in fc layers.
    CNN_weight_reg: Regularizer function (or None as default)
        Specify regularizer for kernel weights in CNN layers.
    CNN_bias_reg: Regularizer function (or None as default)
        Specify regularizer for biases in CNN layers.
    fc_weight_reg: Regularizer function (or None as default)
        Specify regularizer for kernel weights in fc layers.
    fc_bias_reg: Regularizer function (or None as default)
        Specify regularizer for biases in fc layers.
    learning_rate: Float
        Set the learning rate for the optimizer.
    loss_function: String (e.g. 'mse') 
        Choose the loss function.
    
    Returns:
    ========
    compiled TF model
    
    """
    
    # Start model definition:
    model = Sequential()
    
    # Add input layer:
    input_shape = (input_length, n_features)
    model.add(Input(shape=input_shape))
    
    # Add CNN layer(s):
    for i in range(len(CNN_filters)):
        model.add(Conv1D(filters=CNN_filters[i], kernel_size=CNN_kernel_sizes[i], strides=1,
                         kernel_initializer=CNN_weight_init, bias_initializer=CNN_bias_init, 
                         kernel_regularizer=CNN_weight_reg, bias_regularizer=CNN_bias_reg))
        # Optionally add batch normalization:
        if batch_normalization:
            model.add(BatchNormalization())
        
        # Add activation after convolution and optionally - batch normalization:
        model.add(LeakyReLU(alpha=alpha))
        
        # Add max pooling, if input_length exceeds threshold to limit number of trainable parameters:
        if input_length >= max_pooling_threshold:
            model.add(MaxPooling1D(pool_size=2))
    
    # Flatten CNN output:
    model.add(Flatten())
    
    # Add hidden fc layer(s):
    for i in range(len(fc_units)):
        model.add(Dense(units=fc_units[i], activation = fc_activation,
                       kernel_initializer=fc_weight_init, bias_initializer=fc_bias_init,
                       kernel_regularizer=fc_weight_reg, bias_regularizer=fc_bias_reg))
    
    # Add output unit:
    model.add(Dense(units=1, name = "output", activation = output_activation,
                    kernel_initializer=fc_weight_init, bias_initializer=fc_bias_init,
                    kernel_regularizer=fc_weight_reg, bias_regularizer=fc_bias_reg))

    # Compile model with desired loss function:
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=loss_function,
                  metrics=(['mse']))
   
    return model