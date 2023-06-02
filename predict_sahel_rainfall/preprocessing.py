# Tech preambel:
import numpy as np
import pandas as pd


def load_data(
    data_url, ESM, target_index, input_features, add_months, norm_target, lead_time
):
    """
    Function to load CICMoD data as inputs and target index with specified lead time.
    Optionally add months as additional one-hot encoded input feature and normalize target index:

    Parameters:
    ===========
    data_url: str
        Set url to csv file containing CICMoD indices from desired release.
    ESM: str
        Choose ESM from 'CESM' or 'FOCI'
    target_index: str
        Select target index from indices included in CICMoD.
    input_features: array of str
        Select input features.
    add_months: boolean
        Choose, whether to add months as one-hot encoded features.
    norm_target: boolean
        Choose, whether to normalize target index.
    lead_time: int
        Set lead time in months for target index.
        Needs to be positive (> 0).

    Returns:
    ========
    inputs, target:
        Numpy arrays containing inputs and target.

    """

    # Load data:
    climind = pd.read_csv(data_url)

    # Format data:
    climind = climind.set_index(["model", "year", "month", "index"]).unstack(level=-1)[
        "value"
    ]

    # Extract data for specified model, reset index and drop columns year and month:
    climind_ESM = climind.loc[(ESM)].reset_index().drop(columns=["year", "month"])

    # Extract target index and keep ALL indices as inputs:
    target = climind_ESM.loc[:, climind_ESM.columns == target_index]
    inputs = climind_ESM[input_features]

    # If desired, add months as additional one-hot encoded input features:
    if add_months:
        inputs["Jan"] = (
            climind.loc[(ESM)].reset_index().drop(columns=["year"]).loc[:, "month"] == 1
        ).values.astype(int)
        inputs["Feb"] = (
            climind.loc[(ESM)].reset_index().drop(columns=["year"]).loc[:, "month"] == 2
        ).values.astype(int)
        inputs["Mar"] = (
            climind.loc[(ESM)].reset_index().drop(columns=["year"]).loc[:, "month"] == 3
        ).values.astype(int)
        inputs["Apr"] = (
            climind.loc[(ESM)].reset_index().drop(columns=["year"]).loc[:, "month"] == 4
        ).values.astype(int)
        inputs["May"] = (
            climind.loc[(ESM)].reset_index().drop(columns=["year"]).loc[:, "month"] == 5
        ).values.astype(int)
        inputs["Jun"] = (
            climind.loc[(ESM)].reset_index().drop(columns=["year"]).loc[:, "month"] == 6
        ).values.astype(int)
        inputs["Jul"] = (
            climind.loc[(ESM)].reset_index().drop(columns=["year"]).loc[:, "month"] == 7
        ).values.astype(int)
        inputs["Aug"] = (
            climind.loc[(ESM)].reset_index().drop(columns=["year"]).loc[:, "month"] == 8
        ).values.astype(int)
        inputs["Sep"] = (
            climind.loc[(ESM)].reset_index().drop(columns=["year"]).loc[:, "month"] == 9
        ).values.astype(int)
        inputs["Oct"] = (
            climind.loc[(ESM)].reset_index().drop(columns=["year"]).loc[:, "month"]
            == 10
        ).values.astype(int)
        inputs["Nov"] = (
            climind.loc[(ESM)].reset_index().drop(columns=["year"]).loc[:, "month"]
            == 11
        ).values.astype(int)
        inputs["Dec"] = (
            climind.loc[(ESM)].reset_index().drop(columns=["year"]).loc[:, "month"]
            == 12
        ).values.astype(int)

    # If desired, normalize target series to have zero mean and unit variance:
    if norm_target:
        target = (target - np.mean(target)) / np.std(target)

    # Cut target according to desired lead time:
    target = target[lead_time:]

    # Erase last <lead_time> rows from inputs to keep dimensions right:
    inputs = inputs[:-lead_time]

    # Convert inputs and targets to numpy arrays:
    target = np.array(target)
    inputs = np.array(inputs)

    # Return inputs and target:
    return inputs, target


def split_sequence(data, input_length):
    """
    Function to split data into sequences with specified number of time steps as input length:

    Parameters:
    ===========
    data: Numpy array or Pandas DataFrame containing the time series data to be split.
    input_length: Integer number, specifies the number of time steps as input length.

    Returns:
    ========
    Numpy array with data split into sequences.

    """

    X = list()
    for i in range(len(data)):
        # Find the end of this pattern:
        end_ix = i + input_length
        # Check if we are beyond data limits:
        if end_ix > len(data):
            break
        # Gather input and output parts of the pattern:
        seq_x = data[i:end_ix]
        X.append(seq_x)
    return np.array(X)


def train_val_test_split(inputs_split, target_cut, train_test_split, train_val_split):
    """
    Function to split inputs and target into training, validation and test sets.

    Parameters:
    ===========
    inputs_split, target_cut:
        Numpy arrays containing inputs split into sequences of desired input length and accordingly, cut target.
    train_test_split: float
        Specify amount of combined training and validation data relative to test data.
    train_val_split: float
        Specify relative amount of combined training and validation used for training.

    Returns:
    ========
    train_input, val_input, test_input: Numpy arrays
        Training, validation and test inputs.
    train_target, val_target, test_target: Numpy arrays
        Training, validation and test targets.

    """

    # Get number of combined training and validation samples:
    n_train_val = int(train_test_split * len(inputs_split))

    # Split inputs and targets to separate test data from combined training and validation data:
    train_val_input = inputs_split[:n_train_val]
    test_input = inputs_split[n_train_val:]
    train_val_target = target_cut[:n_train_val]
    test_target = target_cut[n_train_val:]

    # Get number of training samples:
    n_train = int(train_val_split * len(train_val_input))

    # Split combined training and validation inputs and targets into training and validation data:
    train_input = train_val_input[:n_train]
    val_input = train_val_input[n_train:]
    train_target = train_val_target[:n_train]
    val_target = train_val_target[n_train:]

    # Return inputs and targets:
    return train_input, train_target, val_input, val_target, test_input, test_target


def scale_norm_inputs(scale_norm, train_input, val_input, test_input, input_features):
    """
    Function to optionally scale or normalize inputs.

    Parameters:
    ===========
    scale_norm: str
        Choose to scale or normalize input features according to statistics from training data:
        'no': Keep raw input features.
        'scale_01': Scale input features with min/max scaling to [0,1].
        'scale_11': Scale input features with min/max scaling to [-1,1].
        'norm': Normalize input features, hence subtract mean and divide by std dev.
    train_input, val_input, test_input:
        Numpy arrays containing training, validation and test inputs.
    input_features: array of str
        Select input features.

    Returns:
    ========
    train_input_scaled, val_input_scaled, test_input_scaled: Numpy arrays
        Optionally scaled or normalized (or unchanged) training, validation and test inputs.
    train_mean, train_std, train_min, train_max: arrays of float
        Statistics from training data.

    """

    ## Get statistics from training data:

    # Get mean, std dev, min and max for all input features, except optionally added one-hot encoded months
    # from training data:
    train_mean = np.mean(train_input[:, :, : len(input_features)], axis=(0, 1))
    train_std = np.std(train_input[:, :, : len(input_features)], axis=(0, 1))
    train_min = np.min(train_input[:, :, : len(input_features)], axis=(0, 1))
    train_max = np.max(train_input[:, :, : len(input_features)], axis=(0, 1))

    ## Optionally scale or normalize input features as specified:

    # 'scale_01': Scale input features with min/max scaling to [0,1].
    if scale_norm == "scale_01":
        # Copy unscaled inputs as initialization for scaled inputs:
        train_input_scaled = np.copy(train_input)
        val_input_scaled = np.copy(val_input)
        test_input_scaled = np.copy(test_input)

        # Apply scaling to all features, except optionally added one-hot encoded months:
        train_input_scaled[:, :, : len(input_features)] = (
            train_input_scaled[:, :, : len(input_features)] - train_min
        ) / (train_max - train_min)
        val_input_scaled[:, :, : len(input_features)] = (
            val_input_scaled[:, :, : len(input_features)] - train_min
        ) / (train_max - train_min)
        test_input_scaled[:, :, : len(input_features)] = (
            test_input_scaled[:, :, : len(input_features)] - train_min
        ) / (train_max - train_min)

    # 'scale_11': Scale input features with min/max scaling to [-1,1].
    if scale_norm == "scale_11":
        # Copy unscaled inputs as initialization for scaled inputs:
        train_input_scaled = np.copy(train_input)
        val_input_scaled = np.copy(val_input)
        test_input_scaled = np.copy(test_input)

        # Apply scaling to all features, except optionally added one-hot encoded months:
        train_input_scaled[:, :, : len(input_features)] = (
            2
            * (train_input_scaled[:, :, : len(input_features)] - train_min)
            / (train_max - train_min)
            - 1
        )
        val_input_scaled[:, :, : len(input_features)] = (
            2
            * (val_input_scaled[:, :, : len(input_features)] - train_min)
            / (train_max - train_min)
            - 1
        )
        test_input_scaled[:, :, : len(input_features)] = (
            2
            * (test_input_scaled[:, :, : len(input_features)] - train_min)
            / (train_max - train_min)
            - 1
        )

    # 'norm': Normalize input features, hence subtract mean and divide by std dev.
    if scale_norm == "norm":
        # Copy unscaled inputs as initialization for scaled inputs:
        train_input_scaled = np.copy(train_input)
        val_input_scaled = np.copy(val_input)
        test_input_scaled = np.copy(test_input)

        # Normalize all features, except optionally added one-hot encoded months:
        train_input_scaled[:, :, : len(input_features)] = (
            train_input_scaled[:, :, : len(input_features)] - train_mean
        ) / train_std
        val_input_scaled[:, :, : len(input_features)] = (
            val_input_scaled[:, :, : len(input_features)] - train_mean
        ) / train_std
        test_input_scaled[:, :, : len(input_features)] = (
            test_input_scaled[:, :, : len(input_features)] - train_mean
        ) / train_std

    # else: Keep raw input features.

    # Return scaled inputs and statistics from training data:
    return (
        train_input_scaled,
        val_input_scaled,
        test_input_scaled,
        train_mean,
        train_std,
        train_min,
        train_max,
    )


def prepare_inputs_and_target(
    data_url,
    ESM,
    target_index,
    input_features,
    add_months,
    norm_target,
    lead_time,
    input_length,
    train_test_split,
    train_val_split,
    scale_norm,
):
    """
    Function to wrap up the complete preprocessing pipeline:
    Load CICMoD data as inputs and target index with specified lead time.
    Optionally add months as additional one-hot encoded input feature and normalize target index.
    Split data into sequences with specified number of time steps.
    Split inputs and target into training, validation and test sets.
    Optionally scale or normalize inputs.

    Parameters:
    ===========
    data_url: str
        Set url to csv file containing CICMoD indices from desired release.
    ESM: str
        Choose ESM from 'CESM' or 'FOCI'
    target_index: str
        Select target index from indices included in CICMoD.
    input_features: array of str
        Select input features.
    add_months: boolean
        Choose, whether to add months as one-hot encoded features.
    norm_target: boolean
        Choose, whether to normalize target index.
    lead_time: int
        Set lead time in months for target index.
        Needs to be positive (> 0).
    input_length: int
        Specify the number of time steps as input length.
    train_test_split: float
        Specify amount of combined training and validation data relative to test data.
    train_val_split: float
        Specify relative amount of combined training and validation used for training.
    scale_norm: str
        Choose to scale or normalize input features according to statistics from training data:
        'no': Keep raw input features.
        'scale_01': Scale input features with min/max scaling to [0,1].
        'scale_11': Scale input features with min/max scaling to [-1,1].
        'norm': Normalize input features, hence subtract mean and divide by std dev.

    Returns:
    ========
    train_input, val_input, test_input: Numpy arrays
        Optionally scaled or normalized training, validation and test inputs, split into sequences of desired input length.
    train_target, val_target, test_target: Numpy arrays
        Corresponding targets, optionally normalized.
    train_mean, train_std, train_min, train_max: arrays of float
        Statistics from training data.

    """

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

    # Split inputs into sequences of specified input length:
    inputs_split = split_sequence(data=inputs, input_length=input_length)

    # Adjust target: Cut first (input_length - 1) entries
    target_cut = target[input_length - 1 :]

    # Split inputs and targets into training, validation and test sets:
    (
        train_input,
        train_target,
        val_input,
        val_target,
        test_input,
        test_target,
    ) = train_val_test_split(
        inputs_split=inputs_split,
        target_cut=target_cut,
        train_test_split=0.9,
        train_val_split=0.8,
    )

    # Optionally scale or normalize inputs:
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

    # Return:
    return (
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
    )
