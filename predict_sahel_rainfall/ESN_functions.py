#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Version: 05 (2022-03-22)

Content: Useful functions / class for EchoStateNetworks (ESNs)

- class ESN
- setESN
- trainESN
- predESN
"""

# Libraries
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, concatenate
import tensorflow.keras.initializers as tfi


### Define custom ESN layer, extending existing tensorflow class Layer

class ESN(tf.keras.layers.Layer):
    
    """
    class ESN extends existing tensorflow class Layer
    
    Called with parameters for initialization:
    ==========================================
    n_res: Number of reservoir units in this layer
    W_in_lim: Determines the range for initialization of input weights W_in and bias b_in and reservoir 
              weights W_res and bias b_res, drawn with RandomUniform in [-W_in_lim,+W_in_lim].
    leak_rate: Used in reservoir state transition
    leak_rate_first_step_YN: If True, use multiplication with alpha already for calculating first 
                             time step's reservoir states.
    leaky_integration_YN: If True, multiply previous time steps' reservoir states with (1-a).
                          If False, omit multiplication with (1-a) in reservoir state transition. 
                          But in any case still multiply new time steps' input (and reservoir recurrence) 
                          with leakrate a after activation.                              
    activation (from ['tanh', 'sigmoid', 'ReLU']): Choose activation function to be used in reservoir 
                                                   state transition.
    
    Function output:
    ================
    Returns Tensor X with all reservoir states for all samples for all times teps, 
      shape: (samples, time steps, n_res)
    Returns Tensor X_T with all FINAL reservoir states for all samples, shape: (samples, n_res)
    """
    
    def __init__(self, n_res, W_in_lim=1, leak_rate=0.5, leak_rate_first_step_YN=True, leaky_integration_YN=True,
                 activation='tanh', verbose=0):
        super(ESN, self).__init__()
        
        # Setup reservoir units
        self.n_res = n_res
        self.W_in_lim = W_in_lim
        self.leak_rate = leak_rate
        self.leak_rate_first_step_YN = leak_rate_first_step_YN
        self.leaky_integration_YN = leaky_integration_YN
        self.activation = activation
        self.res_units_init = tf.keras.layers.Dense(units=n_res, 
                                                    activation=None, 
                                                    use_bias=True,
                                                    kernel_initializer=tfi.RandomUniform(minval=-W_in_lim, maxval=W_in_lim, seed=None),
                                                    bias_initializer=tfi.RandomUniform(minval=-W_in_lim, maxval=W_in_lim, seed=None))
        self.res_units = tf.keras.layers.Dense(units=n_res, 
                                               activation=None, 
                                               use_bias=True,
                                               kernel_initializer=tfi.RandomUniform(minval=-W_in_lim, maxval=W_in_lim, seed=None),
                                               bias_initializer=tfi.RandomUniform(minval=-W_in_lim, maxval=W_in_lim, seed=None))
        self.verbose=verbose
    
    def call(self, inputs):
    
        # Connect inputs to reservoir units to get initial reservoir state (t=1), called x_prev, since
        # it will be used as "previous" state when calculating further reservoir states.
        
        # Apply desired activation function for reservoir state transition: 'tanh' or 'sigmoid'
        if self.activation=='tanh':
            
            # Optionally omit multiplication with alpha for calculating first timestep's reservoir states:
            if self.leak_rate_first_step_YN:
                x_prev = self.leak_rate * tf.tanh(self.res_units_init(inputs[:,0:1,:])) # x(1) = leak_rate * tanh(W_in * u(1))
            else:
                x_prev = tf.tanh(self.res_units_init(inputs[:,0:1,:])) # x(1) = tanh(W_in * u(1))
        
        elif self.activation=='sigmoid':
            
            # Optionally omit multiplication with alpha for calculating first timestep's reservoir states:
            if self.leak_rate_first_step_YN:
                x_prev = self.leak_rate * tf.keras.activations.sigmoid(self.res_units_init(inputs[:,0:1,:])) # x(1) = leak_rate * sigm(W_in * u(1))
            else:
                x_prev = tf.sigmoid(self.res_units_init(inputs[:,0:1,:])) # x(1) = sigm(W_in * u(1))
                
        elif self.activation=='ReLU':
            
            # Optionally omit multiplication with alpha for calculating first timestep's reservoir states:
            if self.leak_rate_first_step_YN:
                x_prev = self.leak_rate * tf.keras.activations.relu(self.res_units_init(inputs[:,0:1,:])) # x(1) = leak_rate * sigm(W_in * u(1))
            else:
                x_prev = tf.keras.activations.relu(self.res_units_init(inputs[:,0:1,:])) # x(1) = sigm(W_in * u(1))
        
        # Initialize storage X for all reservoir states (samples, timesteps, n_res):
        # Store x_prev as x_1 in X
        X = x_prev
        
        # Now loop over remaining time steps t = 2..T
        T = inputs.shape[1]
        
        for t in range(1,T):
            
            # Case distinction according to leaky_integration_YN:
            # Either have leaky integration reservoir:
            if self.leaky_integration_YN:
                
                # Considered desired activation function for state transition:
                if self.activation=='tanh':
                    x_t = (1 - self.leak_rate) * x_prev + self.leak_rate * tf.tanh(self.res_units_init(inputs[:,t:t+1,:]) + self.res_units(x_prev))
                elif self.activation=='sigmoid':
                    x_t = (1 - self.leak_rate) * x_prev + self.leak_rate * tf.keras.activations.sigmoid(self.res_units_init(inputs[:,t:t+1,:]) + self.res_units(x_prev))
                elif self.activation=='ReLU':
                    x_t = (1 - self.leak_rate) * x_prev + self.leak_rate * tf.keras.activations.relu(self.res_units_init(inputs[:,t:t+1,:]) + self.res_units(x_prev))
        
            # Or have non-leaky integration reservoir:
            else:
                
                # Considered desired activation function for state transition:
                if self.activation=='tanh':
                    x_t = x_prev + self.leak_rate * tf.tanh(self.res_units_init(inputs[:,t:t+1,:]) + self.res_units(x_prev))
                elif self.activation=='sigmoid':
                    x_t = x_prev + self.leak_rate * tf.keras.activations.sigmoid(self.res_units_init(inputs[:,t:t+1,:]) + self.res_units(x_prev))
                elif self.activation=='ReLU':
                    x_t = x_prev + self.leak_rate * tf.keras.activations.relu(self.res_units_init(inputs[:,t:t+1,:]) + self.res_units(x_prev))
                    
            # Store x_t in X
            X = tf.concat([X, x_t], axis=1)
            
            # x_t becomes x_prev for next timestep
            x_prev = x_t
        
        # Return both: ALL reservoir states X and final reservoir states X[T].
        return X, X[:,-1,:]


### Define function setESN to set up ESN model.

def setESN(input_length, in_features, out_features, n_layers, n_res, W_in_lim, leak_rate, leak_rate_first_step_YN,
           leaky_integration_YN, activation, spec_radius, sparsity, verbose=False):
    
    """
    function setESN to set up ESN model
    
    Sets up an ESN model with desired number of ESN layers. Then modifies reservoir weights for all ESN layers,
    to fulfill desired properties according to specified spectral radius.
    
    Input parameters:
    =================
    input_length (int): Specified number of time steps per input sample.
    in_features (int): Number of input features, e.g. original series plus decomposed parts L, S and R --> 4
    out_features (int): Number of output features
    n_layers (int): Number of ESN layers in the model.
    n_res (int): Number of reservoir units.
    W_in_lim (float): Initialize input weights from random uniform distribution in [- W_in_lim, + W_in_lim]
    leak_rate (float): Leak rate used in transition function of reservoir states.
    leak_rate_first_step_YN: If True, use multiplication with alpha already for calculating first 
                             time step's reservoir states.
    leaky_integration_YN: If True, multiply previous time steps' reservoir states with (1-a).
                          If False, omit multiplication with (1-a) in reservoir state transition. 
                          But in any case still multiply new time steps' input (and reservoir recurrence) 
                          with leakrate a after activation.     
    spec_radius (float): Spectral radius, becomes largest Eigenvalue of reservoir weight matrix
    sparsity (float): Sparsity of reservoir weight matrix.
    
    Function output:
    ================
    Returns complete model "model".
    Returns short model "model_short" without output layer, for getting final reservoir states for given inputs.
    Returns model "all_states" without output layer, for getting reservoir states 
    for ALL time steps for given inputs.
    
    """
    
    ## Set up model
    
    # Input layer
    model_inputs = Input(shape=(input_length, in_features)) # (time steps, input features)
    
    # Set up storage for layers' final reservoir state tensors:    
    X_T_all = []
    
    ## Loop for setting up desired number of ESN layers:
    for l in range(n_layers):
        
        # First ESN needs to be connected to model_inputs:
        if l == 0:
            
            # Use custom layer for setting up reservoir, returns ALL reservoir states X and FINAL reservoir states X_T.
            X, X_T = ESN(n_res=n_res, W_in_lim=W_in_lim, leak_rate=leak_rate,
                         leak_rate_first_step_YN=leak_rate_first_step_YN, leaky_integration_YN=leaky_integration_YN, 
                         activation=activation)(model_inputs)
            
            # Store resulting final reservoir states:            
            X_T_all.append(X_T)
            
        # Further ESN layers need to be connected to previous ESN layer:
        else:
            
            # Use new custom layer for setting up reservoir, again returns ALL reservoir states X and 
            # FINAL reservoir states X_T.
            X, X_T = ESN(n_res=n_res, W_in_lim=W_in_lim, leak_rate=leak_rate,
                         leak_rate_first_step_YN=leak_rate_first_step_YN, leaky_integration_YN=leaky_integration_YN, 
                         activation=activation)(X)
            
            # Store resulting final reservoir states:            
            X_T_all.append(X_T)
            
    ## Concatenate final reservoir states from ALL layers before passing result to output layer:

    # In case we only have ONE layer, no concatenation is required:
    if n_layers == 1:
        X_T_concat = X_T_all[0]

    # Else concatenate stored final reservoir states using lambda-function:
    else:
        X_T_concat = Lambda(lambda x: concatenate(x, axis=-1))(X_T_all)

    # Output unit
    output = Dense(units=out_features, activation=None, use_bias=True, 
                   kernel_initializer=tfi.RandomUniform(minval=-W_in_lim, maxval=W_in_lim, seed=None),
                   bias_initializer=tfi.RandomUniform(minval=-W_in_lim, maxval=W_in_lim, seed=None),
                   name='output')(X_T_concat)

    # Define complete model "model" plus short model "model_short" omitting the output layer, 
    # for getting reservoir states for given inputs.
    model = Model(model_inputs, output, name='model')
    model_short = Model(model_inputs, X_T_concat, name='model_short')
    
    # Define another shortened model "all_states" to get all reservoir states X from last ESN layer:
    all_states = Model(model_inputs, X, name='all_states')    
    
    ## Modify reservoir weights W_res using spectral radius:

    # Get model weights for ALL layers
    model_weights = np.array(model.get_weights())

    # Loop over desired number of ESN layers for reservoir weights:
    for l in range(n_layers):

        # Extract reservoir weights
        W_res = model_weights[2 + (l * 4)]

        # Need temporary matrix W_temp to implement sparsity manually
        W_temp = np.random.uniform(low=0, high=1, size=(n_res,n_res))
        W_sparse = W_temp <= sparsity

        # Now apply sparsity to initial W_res
        W = W_sparse * W_res

        # Get largest Eigenvalue of W
        ev_max = np.max(np.real(np.linalg.eigvals(W)))

        # Finally set up W_res
        W_res = spec_radius * W / ev_max

        # Integrate modified reservoir weights back into model weights
        
        # Extract reservoir weights
        model_weights[2 + (l * 4)] = W_res

    # Get modified reservoir weights for all ESN layers back into the model
    model.set_weights(model_weights)
    
    
    # Optionally reveal model summaries proof of sparsity and max. Eigenvalues for reservoir weights
    if verbose:
        
        # Print model summaries
        model.summary()
        model_short.summary()  
        
       # Check sparsity and max Eigenvalues for ALL ESN layers' reservoir weights:
        # Get model weights for ALL layers
        model_weights = np.array(model.get_weights())

        # Loop over layers:
        for l in range(n_layers):
            W_res = model_weights[2 + (l * 4)]            

            print("\nLayer ", l+1)
            print("========")
            print("W_res sparsity: ", sum(sum(W_res != 0)) / (W_res.shape[0]**2))
            #print("W_res max EV: ", np.max(np.real((np.linalg.eigvals(W_res)))))

    # Return models
    return model, model_short, all_states


### Define function trainESN to train output weights and bias for an already set up ESN model.

def trainESN(model, model_short, train_input, train_target, verbose=False):
    
    """
    function trainESN to train output weights and bias for an already set up ESN model
    
    Input parameters:
    =================
    model: complete ESN model, as returned from e.g. setESN.
    model_short: Short model as provided by e.g. setESN, without output layer, 
                 for getting reservoir states for given inputs.
    train_input: Input samples (samples, timesteps, input features) to be used for training.
    train_target: True targets for train inputs (samples, output features).
    verbose (True/False)): if True, plot histogram of trained output weights and give additionala information
                           on bias after training
    
    Function output:
    ================
    Returns complete model "model" with trained output weights and bias.
    """
    
    # Get number of output features from train targets:
    out_features = train_target.shape[1]

    # Get final reservoir states for all train samples from short model    
    X_T_train = model_short.predict(train_input)

    # Extract output weights and bias.
    # Note: output layer is the LAST layer of the model, find weights and bias at position "-2" and "-1", respectively.
    model_weights = np.array(model.get_weights())
    W_out = model_weights[-2]
    b_out = model_weights[-1]
    
    # Create vector of shape (samples, 1) containing ONEs to be added as additional column to final reservoir states.
    X_add = np.ones((X_T_train.shape[0], 1))

    # Now add vector of ONEs as additional column to final reservoir states X_T_train.
    X_T_train_prime = np.concatenate((X_T_train, X_add), axis=-1)

    # Then need pseudo-inverse of final reservoir states in augmented notation
    X_inv_prime = np.linalg.pinv(X_T_train_prime)

    # Optionally reveal details on dimensions:
    if verbose:
        
        print("\nshape of train input (samples, timesteps, input features): ", train_input.shape)
        print("shape of model output X_T (samples, n_res): ", X_T_train.shape)
  
        print("\nFinal reservoir states in augmented notation, shape: ", X_T_train_prime.shape)
        print("\ntrain_target shape (samples, output features): ", train_target.shape)        
    
        print("\nW_out shape: ", W_out.shape)
        print("b_out shape: ", b_out.shape)
    
    # Need to train output features seperately, hence loop over number of output features:
    for out_feature in range(out_features):    

        # Then get output weights, in augmented notation
        W_out_prime = np.matmul(X_inv_prime, train_target[:,out_feature:out_feature+1])

        # Now split output weights in augmented notation into trained output weights W_out and output bias b_out.
        W_out = W_out_prime[:-1,:]
        b_out = W_out_prime[-1:,0]
        #print("\nW_out: \n", W_out)

        # Integrate trained output weights and bias into model weights
        model_weights[-2][:,out_feature:out_feature+1] = W_out
        model_weights[-1][out_feature] = b_out
        model.set_weights(model_weights)
    
        # Optionally reveal details on traines output weights and bias(es)
        if verbose:

            print("\noutput feature ", out_feature, ", trained b_out: ", b_out)

            # Plot histogram of trained output weights
            nBins = 100
            fig, axes = plt.subplots(1, 1, figsize=(10,5))
            axes.hist(W_out[:,0], nBins, color="blue")
            axes.set_ylabel("counts")
            axes.set_title("Histogram of trained output weights")
            plt.show()

    return model


### Define function predESN to evaluate performance of trained ESN model.

def predESN(model, train_input, val_input, train_target, val_target, verbose=False):

    """
    function predESN to evaluate performance of trained ESN model
    
    Input parameters:
    =================
    model: complete ESN model, as returned from e.g. setESN.
    train_input: Input samples (samples, time steps, input features) to be used for training.
    val_input: Input samples (samples, time steps, input features) to be used for validation.
    train_target: True targets for train inputs (samples, output features).
    val_target: True targets for validation inputs (samples, output features).
    verbose (True/False)): if True, reveal details on model performance and show fidelity plots.
    
    Function output:
    ================
    Returns model predictions on train and validation inputs.
    Returns evaluation metrics 'mean-absolute-error' (mae) and 'mean-squared-error' (mse) 
      on train and val. data.
    """
    
    ## Get predictions from "long" model on train and validation input
    val_pred = model.predict(val_input)
    train_pred = model.predict(train_input)
    
    # Calculate mean-absolute and mean-squared error of model predictions compared to targets:
    train_mae = np.round(sum(np.abs(train_target[:,0] - train_pred[:,0])) / len(train_target), 4)
    val_mae = np.round(sum(np.abs(val_target[:,0] - val_pred[:,0])) / len(val_target), 4)
    train_mse = np.round(sum((train_target[:,0] - train_pred[:,0])**2) / len(train_target), 4)
    val_mse = np.round(sum((val_target[:,0] - val_pred[:,0])**2) / len(val_target), 4)
    
    # Optionally reveal model summaries proof of sparsity and max. Eigenvalues for reservoir weights
    if verbose:
        
        print("\nshape of val input (samples, time steps, features): ", val_input.shape)
        print("shape of train input (samples, time steps, features): ", train_input.shape)

        print("\nshape of model predictions on validation input (samples, 1): ", val_pred.shape)
        print("shape of val targets (samples, 1): ", val_target.shape)

        print("\ntrain_mae: ", train_mae)
        print("val_mae: ", val_mae)
        
        print("\ntrain_mse: ", train_mse)
        print("val_mse: ", val_mse)
        
        # Fidelity check: Plot train_pred vs. train_targets
        plt.figure(figsize=(16,8))
        plt.plot(range(len(train_target)),train_target,'b',label="true data", alpha=0.3)
        plt.plot(range(len(train_pred)),train_pred,'k',  alpha=0.8, label='pred ESN')
        plt.title('Fidelity check on TRAIN data', fontsize=16)
        plt.xlabel('timestep', fontsize=14)
        plt.ylabel('target', fontsize=14)
        plt.legend(fontsize=14)
        plt.show()

        # Fidelity check: Plot val_pred vs. val_targets
        plt.figure(figsize=(16,8))
        plt.plot(range(len(val_target)),val_target,'b',label="true data", alpha=0.3)
        plt.plot(range(len(val_pred)),val_pred,'k',  alpha=0.8, label='pred ESN')
        plt.title('Fidelity check on VALIDATION data', fontsize=16)
        plt.xlabel('timestep', fontsize=14)
        plt.ylabel('target', fontsize=14)
        plt.legend(fontsize=14)
        plt.show()
        
    return train_pred, val_pred, train_mae, val_mae, train_mse, val_mse
