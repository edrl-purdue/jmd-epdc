# Import packages
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
import gpflow as gpf

def build_GPR_model_EPDC(X_train_n, y_train_n):
    # Dimension of the input space
    n_dimensions = X_train_n.shape[1]

    # Kernel for each output
    kernel = gpf.kernels.Matern32(lengthscales=np.ones(n_dimensions)) + gpf.kernels.Constant()

    # Instantiate GPR models
    m = gpf.models.GPR((X_train_n, y_train_n), kernel)

    # Lower and upper bounds for characteristic lenghts of kernels
    lb_sig = np.float64(0.01 * np.ones(n_dimensions))
    ub_sig = np.float64(10.0 * np.ones(n_dimensions))

    m.kernel.kernels[0].lengthscales.assign((lb_sig + ub_sig) / 2)

    # Fix upper and lower bounds of hyperparameters (lengthscales)
    tfb = tfp.bijectors
    sigmoid_transf = tfb.Sigmoid(low=lb_sig, high=ub_sig)
    m_temp = m.kernel.kernels[0].lengthscales
    m.kernel.kernels[0].lengthscales = gpf.Parameter(m_temp.numpy(), transform=sigmoid_transf)

    # Fix noise-variance to a low value to perform interpolation
    pert_tol1 = 0.0001
    m.likelihood.variance.assign(pert_tol1)
    gpf.set_trainable(m.likelihood.variance, False)

    return m

def build_GPR_model(X_train_n, y_train_n):
    # Dimension of the input space
    n_dimensions = X_train_n.shape[1]

    # Kernel for each output
    kernel = gpf.kernels.Matern52(lengthscales=np.ones(n_dimensions)) + gpf.kernels.Constant()

    # Instantiate GPR models
    m = gpf.models.GPR((X_train_n, y_train_n), kernel)

    # Lower and upper bounds for characteristic lenghts of kernels
    lb_sig = np.float64(0.01 * np.ones(n_dimensions))
    ub_sig = np.float64(10.0 * np.ones(n_dimensions))

    m.kernel.kernels[0].lengthscales.assign((lb_sig + 1) / 2)

    # Fix upper and lower bounds of hyperparameters (lengthscales)
    tfb = tfp.bijectors
    sigmoid_transf = tfb.Sigmoid(low=lb_sig, high=ub_sig)
    m_temp = m.kernel.kernels[0].lengthscales
    m.kernel.kernels[0].lengthscales = gpf.Parameter(m_temp.numpy(), transform=sigmoid_transf)

    # Fix noise-variance to a low value to perform interpolation
    pert_tol1 = 0.0001
    m.likelihood.variance.assign(pert_tol1)
    gpf.set_trainable(m.likelihood.variance, False)

    return m

def build_GPC_model(X_train_n, fc_train):
    # Dimension of the input space
    n_dimensions = X_train_n.shape[1]

    # Kernel for each output
    kernel = gpf.kernels.Matern12(lengthscales=np.ones(n_dimensions)) + gpf.kernels.Constant()

    # Instantiate GPR models
    m = gpf.models.VGP((X_train_n, fc_train), kernel = kernel, likelihood=gpf.likelihoods.Bernoulli())

    # Lower and upper bounds for characteristic lenghts of kernels
    lb_sig = np.float64(0.01 * np.ones(n_dimensions))
    ub_sig = np.float64(1.0 * np.ones(n_dimensions))

    m.kernel.kernels[0].lengthscales.assign((lb_sig + 1) / 2)

    # Fix upper and lower bounds of hyperparameters (lengthscales)
    tfb = tfp.bijectors
    sigmoid_transf = tfb.Sigmoid(low=lb_sig, high=ub_sig)
    m_temp = m.kernel.kernels[0].lengthscales
    m.kernel.kernels[0].lengthscales = gpf.Parameter(m_temp.numpy(), transform=sigmoid_transf)

    return m

def train_gp_model_EPDC(m, X_train_n, f_train, NaN_flag, iter):
    # Try to train the GP model using L-BFGS-B optimizer
    opt = gpf.optimizers.Scipy()
    opt.minimize(m.training_loss, variables=m.trainable_variables, options=dict(maxiter=1000),
                    method="L-BFGS-B")
    
    # Check for nan values; if they occur reset model and use adagrad to train GPs
    if np.isnan(m.kernel.kernels[0].lengthscales.numpy()).any():
        m = build_GPR_model_EPDC(X_train_n, f_train)
        optimizer = tf.keras.optimizers.Adagrad(learning_rate=0.01)
        print('Training GP model using adagrad')

        @tf.function
        def step_1(i):
            optimizer.minimize(m.training_loss, m.trainable_variables)
        for i in tf.range(10000):
            step_1(i)
        NaN_flag[iter] = 1
    # print('Training GPR model completed')
    return m, NaN_flag