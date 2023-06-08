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

import pytest
from numpy.testing import assert_almost_equal

# Import functions to test:
from predict_sahel_rainfall.models import (
    set_CNN_fc
)


# Set parameters for test:
input_length=24
n_features=10
CNN_filters=[10,20]
CNN_kernel_sizes=[5,5]
batch_normalization=True
alpha=0.3
fc_units=[20,10]
fc_activation='sigmoid'
output_activation='linear'
CNN_weight_init='glorot_uniform'
CNN_bias_init='zeros'
fc_weight_init='glorot_uniform'
fc_bias_init='zeros'
CNN_weight_reg=None
CNN_bias_reg=None
fc_weight_reg=None
fc_bias_reg=None
learning_rate=0.0001
loss_function='mse'


@pytest.mark.parametrize("max_pooling_threshold", list([20, 30]))
def test_pooling_threshold(max_pooling_threshold):
    """Test, if function sets up model with correct number of layers, with or without pooling layer, according to threshold."""

    # Set up CNN/fc model:
    model = set_CNN_fc(
        input_length=input_length, 
        n_features=n_features, 
        max_pooling_threshold=max_pooling_threshold,
        CNN_filters=CNN_filters, 
        CNN_kernel_sizes=CNN_kernel_sizes, 
        batch_normalization=batch_normalization,
        alpha=alpha,
        fc_units=fc_units, 
        fc_activation=fc_activation,
        output_activation=output_activation,
        CNN_weight_init=CNN_weight_init,
        CNN_bias_init=CNN_bias_init,
        fc_weight_init=fc_weight_init,
        fc_bias_init=fc_bias_init,
        CNN_weight_reg=CNN_weight_reg,
        CNN_bias_reg=CNN_bias_reg,
        fc_weight_reg=fc_weight_reg,
        fc_bias_reg=fc_bias_reg,
        learning_rate=learning_rate, 
        loss_function=loss_function
    )

    # Test, number of model layers includes pooling layers, if input lenght exceeds threshold:
    if input_length >= max_pooling_threshold:
        assert len(model.layers) == 12
    else:
        assert len(model.layers) == 10
