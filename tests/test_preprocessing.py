# Tech preamble:
import numpy as np
import pandas as pd
import pytest

from numpy.testing import assert_almost_equal

# Import functions to test:
from predict_sahel_rainfall.preprocessing import (
    get_target_months,
    load_data,
    prepare_inputs_and_target,
    scale_norm_inputs,
    split_sequence,
    train_val_test_split,
)


# Create dummy array:
dummy_array = np.array(
    [
        [1.0, 11.0],
        [2.0, 12.0],
        [3.0, 13.0],
        [4.0, 14.0],
        [5.0, 15.0],
        [6.0, 16.0],
        [7.0, 17.0],
        [8.0, 18.0],
        [9.0, 19.0],
        [10.0, 20.0],
    ]
)

# Set parameters for test:
data_url = "https://github.com/MarcoLandtHayen/climate_index_collection/releases/download/v2023.03.29.1/climate_indices.csv"
target_index = "PREC_SAHEL"
input_features = ["PREC_SAHEL", "SAM_ZM"]
add_months = True
norm_target = True
lead_time = 1
input_length = 3
train_test_split = 0.9
train_val_split = 0.8


@pytest.mark.parametrize("ESM", list(["CESM", "FOCI"]))
def test_get_target_months(ESM):
    """Test, if function correctly extracts months according to targets."""

    # Get target months for current ESM and lead time:
    (train_months, val_months, test_months) = get_target_months(
        data_url=data_url,
        ESM=ESM,
        lead_time=lead_time,
        input_length=input_length,
        train_test_split=train_test_split,
        train_val_split=train_val_split,
    )

    # Test, if first months correctly takes lead time and input length into account:
    assert train_months[0] == (lead_time + input_length) % 12


@pytest.mark.parametrize("ESM", list(["CESM", "FOCI"]))
def test_load_data(ESM):
    """Test, if function correctly loads test csv."""

    # Load data:
    inputs, target = load_data(
        data_url=data_url,
        ESM=ESM,
        target_index=target_index,
        input_features=input_features,
        add_months=add_months,
        norm_target=norm_target,
        lead_time=lead_time,
    )

    # Test, if months are added as additional one-hot encoded input features:
    assert inputs.shape[1] == len(input_features) + 12

    # Test, if we end up with only single target feature
    assert target.shape[1] == 1


def test_split_sequence():
    """Test, if function correctly splits data into sequences of specified input length."""

    # Split dummy array into sequences of specified input length:
    inputs_split = split_sequence(data=dummy_array, input_length=input_length)

    # Test, if number of resulting sequences is correct:
    assert inputs_split.shape[0] == len(dummy_array) - (input_length - 1)


def test_train_val_test_split():
    """Test, if function correctly splits inputs and target into training, validation and test sets."""

    # Split dummy array into sequences of specified input length:
    dummy_inputs_split = split_sequence(data=dummy_array, input_length=input_length)

    # Create corresponding dummy target:
    dummy_target_cut = dummy_array[input_length - 1 :, 0]

    # Split dummy inputs and target into training, validation and test sets:
    (
        train_input,
        train_target,
        val_input,
        val_target,
        test_input,
        test_target,
    ) = train_val_test_split(
        inputs_split=dummy_inputs_split,
        target_cut=dummy_target_cut,
        train_test_split=train_test_split,
        train_val_split=train_val_split,
    )

    # Test dimensions of training, validation and test inputs and targets:
    assert train_input.shape == (5, 3, 2)
    assert val_input.shape == (2, 3, 2)
    assert test_input.shape == (1, 3, 2)
    assert train_target.shape == (5,)
    assert val_target.shape == (2,)
    assert test_target.shape == (1,)


@pytest.mark.parametrize("scale_norm", list(["scale_01", "scale_11", "norm"]))
def test_scale_norm_inputs(scale_norm):
    """Test, if training inputs are correctly normalized or scaled."""

    # Split dummy array into sequences of specified input length:
    dummy_inputs_split = split_sequence(data=dummy_array, input_length=input_length)

    # Create corresponding dummy target:
    dummy_target_cut = dummy_array[input_length - 1 :]

    # Split dummy inputs and target into training, validation and test sets:
    (
        train_input,
        train_target,
        val_input,
        val_target,
        test_input,
        test_target,
    ) = train_val_test_split(
        inputs_split=dummy_inputs_split,
        target_cut=dummy_target_cut,
        train_test_split=train_test_split,
        train_val_split=train_val_split,
    )
    # Scale or normalize inputs:
    (
        train_input,
        val_input,
        test_input,
        train_mean,
        train_std,
        train_min,
        train_max,
    ) = scale_norm_inputs(
        scale_norm=scale_norm,
        train_input=train_input,
        val_input=val_input,
        test_input=test_input,
        input_features=input_features,
    )

    ## Test scaling or normalization of training inputs:

    # Scale inputs to [0,1]:
    if scale_norm == "scale_01":
        assert np.min(train_input) == 0
        assert np.max(train_input) == 1

    # Scale inputs to [-1,1]:
    if scale_norm == "scale_11":
        assert np.min(train_input) == -1
        assert np.max(train_input) == 1

    # Normalize inputs to have zero mean and unit variance:
    if scale_norm == "norm":
        assert_almost_equal(actual=np.mean(train_input), desired=0, decimal=3)
        assert_almost_equal(actual=np.std(train_input), desired=1, decimal=3)


@pytest.mark.parametrize("ESM", list(["CESM", "FOCI"]))
def test_norm_target(ESM):
    """Test, if complete preprocessing pipeline ends up with normalized targets, if desired."""

    # Prepare inputs and target:
    (
        train_input,
        train_target,
        val_input,
        val_target,
        test_input,
        test_target,
        train_mean,
        train_std,
        train_min,
        train_max,
    ) = prepare_inputs_and_target(
        data_url=data_url,
        ESM=ESM,
        target_index=target_index,
        input_features=input_features,
        add_months=add_months,
        norm_target=norm_target,
        lead_time=lead_time,
        input_length=input_length,
        train_test_split=train_test_split,
        train_val_split=train_val_split,
        scale_norm="no",
    )

    ## Test normalization of combined training, validation and test targets:
    assert_almost_equal(
        actual=np.mean(np.concatenate([train_target, val_target, test_target])),
        desired=0,
        decimal=3,
    )
    assert_almost_equal(
        actual=np.std(np.concatenate([train_target, val_target, test_target])),
        desired=1,
        decimal=3,
    )
