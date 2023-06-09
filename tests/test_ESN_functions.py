# Tech preamble:
import numpy as np
import pytest
import tensorflow as tf
import tensorflow.keras.initializers as tfi

from numpy.testing import assert_almost_equal
from tensorflow.keras.layers import Dense, Input, Lambda, concatenate
from tensorflow.keras.models import Model

# Import functions to test:
from predict_sahel_rainfall.ESN_functions import ESN, setESN


# Set parameters for test:
input_length = 24
n_features = 10
verbose = False
n_layers = 1
n_res = 300
W_in_lim = 0.1
leak_rate = 0.05
leak_rate_first_step_YN = True
leaky_integration_YN = True
activation = "tanh"
spec_radius = 0.8
sparsity = 0.3
out_features = 1


def test_sparsity_ESN():
    """Test, if function sets up ESN with desired sparsity for reservoir connections."""

    ## Set up ESN model:
    # Get complete model (output = target prediction) plus short model (output final reservoir states from all layers)
    # and all_states (= another shortened model that gives reservoir states for ALL time steps for all inputs).
    model, model_short, all_states = setESN(
        input_length=input_length,
        in_features=n_features,
        out_features=out_features,
        n_layers=n_layers,
        n_res=n_res,
        W_in_lim=W_in_lim,
        leak_rate=leak_rate,
        leak_rate_first_step_YN=leak_rate_first_step_YN,
        leaky_integration_YN=leaky_integration_YN,
        activation=activation,
        spec_radius=spec_radius,
        sparsity=sparsity,
        verbose=verbose,
    )

    # Get model weights for all layers
    model_weights = np.array(model.get_weights())

    # Extract reservoir weights:
    W_res = model_weights[2]

    # Test sparsity of reservoir weights:
    assert_almost_equal(
        actual=sum(sum(W_res != 0)) / (W_res.shape[0] ** 2), desired=sparsity, decimal=1
    )
