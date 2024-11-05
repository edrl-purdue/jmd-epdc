import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import gpflow as gpf
from scipy.stats import qmc

def find_new_design_EPDC(GPR_models, EPDC, X_train_n, PFn, eff_length, n_mc_samples, gamma_dominated, gamma_pareto, n_samples_opt, n_dimensions, learning_rate, num_steps_optimizer, Sigmoid_X, scaling_opt):
    # Initialization of search points (multi-point search)
    sampler = qmc.LatinHypercube(d=n_dimensions)
    X_initial = sampler.random(n=n_samples_opt)
    # Initial points
    X_opt = tf.Variable(Sigmoid_X.inverse(X_initial), trainable=True, dtype=tf.float64)

    # Optimization using automatic differentiation
    @tf.function
    def get_loss_and_grads():
        with tf.GradientTape() as tape:
            # Specify the variables we want to track for the following operations
            tape.watch(X_opt)
            # Loss function
            loss = EPDC(X_opt, GPR_models, X_train_n, PFn, Sigmoid_X, eff_length, n_mc_samples, gamma_dominated, gamma_pareto, scaling_opt)[0] # Here scaling_opt is 1 since we need to normalize X_opt  
            # Compute the gradient of the loss wrt the trainable_variables
            grads = tape.gradient(loss, X_opt)
        return loss, grads

    # Optimizer for the AF
    optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)

    # print('Maximizing acquisition function ...')
    for i in range(num_steps_optimizer):
        # Compute loss and gradients
        loss, grads = get_loss_and_grads()
        # Update the training variables
        optimizer.apply_gradients(zip([grads], [X_opt]))
        # print('l =\n',-loss.numpy(), '\ng =\n', grads.numpy(), '\nX_opt =\n',Sigmoid_X(X_opt).numpy())
    # print('Maximizing acquisition function - completed!')

    # Results of the optimization
    # Re-scale X_opt to the [0 - 1] scale
    x_opt = Sigmoid_X(X_opt).numpy()
    # Select point with highest acquisition value
    index_opt = np.argsort(loss.numpy(), axis=0)[0]
    x_new = x_opt[index_opt, :]
    return x_new


def find_new_design_EEI(GPR_models, EEI, X_train_n, PFn, eff_length, n_samples_opt, n_dimensions, learning_rate, num_steps_optimizer, Sigmoid_X, scaling_opt):
    # Initialization of search points (multi-point search)
    sampler = qmc.LatinHypercube(d=n_dimensions)
    X_initial = sampler.random(n=n_samples_opt)
    # Initial points
    X_opt = tf.Variable(Sigmoid_X.inverse(X_initial), trainable=True, dtype=tf.float64)

    # Optimization using automatic differentiation
    @tf.function
    def get_loss_and_grads():
        with tf.GradientTape() as tape:
            # Specify the variables we want to track for the following operations
            tape.watch(X_opt)
            # Loss function
            loss = EEI(X_opt, GPR_models, PFn, Sigmoid_X, X_train_n, eff_length, scaling_opt)[0] # Here scaling_opt is 1 since we need to normalize X_opt  
            # Compute the gradient of the loss wrt the trainable_variables
            grads = tape.gradient(loss, X_opt)
        return loss, grads

    # Optimizer for the AF
    optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)

    # print('Maximizing acquisition function ...')
    for i in range(num_steps_optimizer):
        # Compute loss and gradients
        loss, grads = get_loss_and_grads()
        # Update the training variables
        optimizer.apply_gradients(zip([grads], [X_opt]))
        # print('l =\n',-loss.numpy(), '\ng =\n', grads.numpy(), '\nX_opt =\n',Sigmoid_X(X_opt).numpy())
    # print('Maximizing acquisition function - completed!')

    # Results of the optimization
    # Re-scale X_opt to the [0 - 1] scale
    x_opt = Sigmoid_X(X_opt).numpy()
    # Select point with highest acquisition value
    index_opt = np.argsort(loss.numpy(), axis=0)[0]
    x_new = x_opt[index_opt, :]
    return x_new


def find_new_design_sr_MO_BO_3GP(GP_models, Composed_acquisition_function_penalized, X_train_n, Beta, Slack, y_c_best, eff_length, n_samples_opt, n_dimensions, learning_rate, num_steps_optimizer, Sigmoid_X, scaling_opt):
    # Load GP models
    GPR = GP_models[0] # Regressor asocciated to the augmented_Tchebycheff function
    GPC = GP_models[1] # Classifier asocciated to the Pareto front
    
    # Initialization of search points (multi-point search)
    sampler = qmc.LatinHypercube(d=n_dimensions)
    X_initial = sampler.random(n=n_samples_opt)
    # Initial points
    X_opt = tf.Variable(Sigmoid_X.inverse(X_initial), trainable=True, dtype=tf.float64)

    # Optimization using automatic differentiation
    @tf.function
    def get_loss_and_grads():
        with tf.GradientTape() as tape:
            # Specify the variables we want to track for the following operations
            tape.watch(X_opt)
            # Loss function
            loss = Composed_acquisition_function_penalized(X_opt, GPR, GPC, Beta, Slack, y_c_best, Sigmoid_X, X_train_n, eff_length, scaling_opt)[0] # Here scaling_opt is 1 since we need to normalize X_opt  
            # Compute the gradient of the loss wrt the trainable_variables
            grads = tape.gradient(loss, X_opt)
        return loss, grads

    # Optimizer for the AF
    optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)

    # print('Maximizing acquisition function ...')
    for i in range(num_steps_optimizer):
        # Compute loss and gradients
        loss, grads = get_loss_and_grads()
        # Update the training variables
        optimizer.apply_gradients(zip([grads], [X_opt]))
        # print('l =\n',-loss.numpy(), '\ng =\n', grads.numpy(), '\nX_opt =\n',Sigmoid_X(X_opt).numpy())
    # print('Maximizing acquisition function - completed!')

    # Results of the optimization
    # Re-scale X_opt to the [0 - 1] scale
    x_opt = Sigmoid_X(X_opt).numpy()
    # Select point with highest acquisition value
    index_opt = np.argsort(loss.numpy(), axis=0)[0]
    x_new = x_opt[index_opt, :]
    return x_new


def find_new_design_sr_MO_BO_3GP_no_classifier(GPR, Composed_acquisition_function_penalized_no_classifier, X_train_n, Beta, Slack, eff_length, n_samples_opt, n_dimensions, learning_rate, num_steps_optimizer, Sigmoid_X, scaling_opt):   
    # Initialization of search points (multi-point search)
    sampler = qmc.LatinHypercube(d=n_dimensions)
    X_initial = sampler.random(n=n_samples_opt)
    # Initial points
    X_opt = tf.Variable(Sigmoid_X.inverse(X_initial), trainable=True, dtype=tf.float64)

    # Optimization using automatic differentiation
    @tf.function
    def get_loss_and_grads():
        with tf.GradientTape() as tape:
            # Specify the variables we want to track for the following operations
            tape.watch(X_opt)
            # Loss function
            loss = Composed_acquisition_function_penalized_no_classifier(X_opt, GPR, Beta, Slack, Sigmoid_X, X_train_n, eff_length, scaling_opt)[0] # Here scaling_opt is 1 since we need to normalize X_opt  
            # Compute the gradient of the loss wrt the trainable_variables
            grads = tape.gradient(loss, X_opt)
        return loss, grads

    # Optimizer for the AF
    optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)

    # print('Maximizing acquisition function ...')
    for i in range(num_steps_optimizer):
        # Compute loss and gradients
        loss, grads = get_loss_and_grads()
        # Update the training variables
        optimizer.apply_gradients(zip([grads], [X_opt]))
        # print('l =\n',-loss.numpy(), '\ng =\n', grads.numpy(), '\nX_opt =\n',Sigmoid_X(X_opt).numpy())
    # print('Maximizing acquisition function - completed!')

    # Results of the optimization
    # Re-scale X_opt to the [0 - 1] scale
    x_opt = Sigmoid_X(X_opt).numpy()
    # Select point with highest acquisition value
    index_opt = np.argsort(loss.numpy(), axis=0)[0]
    x_new = x_opt[index_opt, :]
    return x_new