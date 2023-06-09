# Import external packages
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
import gpflow as gpf
import scipy.io
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import qmc
from pymoo.indicators.hv import HV
from pymoo.problems import get_problem

# Import in-house modules for multiobjective optimization
import Test_functions
from optimization_files.sr_MO_BO_3GP_approach import Augmented_Tchebycheff, Composed_acquisition_function_penalized_no_classifier
from optimization_files.vanilla_GP_model import build_GPR_model
from optimization_files.find_pareto import Pareto_front

#%% ######################### Viennet - 2dv ####################################################
#################### Optimization settings (These section can be changed by the designer)
####################################################
test_index = 0 # select a test index from 0 to 4 (initial sampling plan)
total_iter = 100 # Total number of iterations
n_dimensions = 2
num_steps_optimizer = 400  # Number of iterations of SGD-based optimizer (300 - 400)
learning_rate = 0.25  # Learning rate optimizer
n_samples_opt = np.min((25 * n_dimensions, 500)) # Number of points for multi-point optimization of the AF
eff_length = ((0.05 ** 2) * n_dimensions) ** 0.5  # Effective lenght for penalization of similar designs
ref_point = np.array([10.0, 60, 0.2]) # Reference point for calculation of dominated hypervolume
X_bounds = np.concatenate((-3*np.ones((1,n_dimensions)),3*np.ones((1,n_dimensions))),0) # Box constraints of the optimization problem
# Settings sr-MOBO approach
rho_tb = 0.65
lambda_tb = 0.01
Beta = 0.95
Slack = 10
# Update weighting vector according to the number of iterations
weights_dic = scipy.io.loadmat('data/DOE_weights_3_obj_100.mat')
weights_sde = weights_dic.get('w_DOE_samples')

########################################################################################################
################## Optimization approach (Do not change if you want to preserve the methodology presented in the JMD publication)
####################################################
# Import initial sampling plan
X_train_dic = scipy.io.loadmat('data/three_objective_Viennet_2_dv.mat')
X_train_lhs = X_train_dic.get('X_initial')

# Evaluate sampling plan
X_train = X_train_lhs[:, :, test_index]
n_init = X_train.shape[0]
F_train = Test_functions.Viennet(X_train)

# Arrays to track behaviour of optimization approach
HV_optimization = np.zeros((total_iter, 1))
NaN_flag = np.zeros((total_iter, 1))

# Normalization of inputs outside optimization loop as we know its maximum and minimum values
scaler_X = MinMaxScaler()
scaler_X.fit(X_bounds)

# Bayesian optimization loop
for iter in range(total_iter):
    # Sample weights
    weights_tb = weights_sde[iter]
    #%% ############# Find Pareto front and Pareto set
    # Pareto designs: O if dominated, 1 if not dominated
    Pareto_index = Pareto_front(F_train)
    # Pareto front
    PF = F_train[Pareto_index]
    # Pareto designs
    PD = X_train[Pareto_index, :]
    # Create scaler instances
    scaler_Y = MinMaxScaler()
    scaler_Y.fit(F_train)
    # Normalize training data
    F_train_n, X_train_n = scaler_Y.transform(F_train), scaler_X.transform(X_train)

    # Record initial Pareto front and Pareto set
    if iter == 0:
        # Pareto designs
        PF_init = PF
        PD_init = PD

    # Transform data according to Augmented_Tchebycheff function
    F_aug_Tcheb = Augmented_Tchebycheff(X_train_n, F_train_n, weights_tb, rho_tb, lambda_tb)
    F_aug_Tcheb_neg = -F_aug_Tcheb

    #%% ############## GPR model of the negative augmented_Tchebycheff function
    # Build GPR model
    GPR = build_GPR_model(X_train_n, F_aug_Tcheb_neg)

    # Train GPR model
    print('Training GPR model of Augmented Tchebycheff function ...')
    opt1 = gpf.optimizers.Scipy()
    opt1.minimize(GPR.training_loss, variables=GPR.trainable_variables, options=dict(maxiter=1000),
                  method="L-BFGS-B")
    # Check for nan values; if they occur reset model and use adagrad to train GPs
    if np.isnan(GPR.kernel.kernels[0].lengthscales.numpy()).any():
        GPR = build_GPR_model(X_train_n, F_aug_Tcheb_neg)
        optimizer_1 = tf.keras.optimizers.Adagrad(learning_rate=0.01)
        print('Training GPR model using adagrad')

        @tf.function
        def step_1(i):
            optimizer_1.minimize(GPR.training_loss, GPR.trainable_variables)

        for i in tf.range(10000):
            step_1(i)
        NaN_flag[iter, 0] = 1
    print('Training GPR model of Augmented Tchebycheff function completed')

    #%% ############## Optimization of acquisition function
    # Optimization using AD
    # Transformation of variables to enforce box constraints
    lb_X = np.float64(np.zeros(n_dimensions))
    ub_X = np.float64(np.ones(n_dimensions))
    bounds = [lb_X, ub_X]
    Sigmoid_X = tfp.bijectors.Sigmoid(low=lb_X, high=ub_X)

    # Initialization of search points (multi-point search)
    sampler = qmc.LatinHypercube(d=n_dimensions)
    X_initial = sampler.random(n=n_samples_opt)
    # Initial points
    X_opt = tf.Variable(Sigmoid_X.inverse(X_initial), trainable=True, dtype=tf.float64)

    @tf.function
    def get_loss_and_grads():
        with tf.GradientTape() as tape:
            # Specify the variables we want to track for the following operations
            tape.watch(X_opt)
            # Loss function
            loss = Composed_acquisition_function_penalized_no_classifier(X_opt, GPR, Beta, Slack, Sigmoid_X, X_train_n, eff_length, opt=1)[0]
            # Compute the gradient of the loss wrt the trainable_variables
            grads = tape.gradient(loss, X_opt)
        return loss, grads

    # Optimizer for the AF
    optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)

    print('Maximizing acquisition function ...')
    for i in range(num_steps_optimizer):
        # Compute loss and gradients
        loss, grads = get_loss_and_grads()
        # Update the training variables
        optimizer.apply_gradients(zip([grads], [X_opt]))
        # print('l =\n',-loss.numpy(), '\ng =\n', grads.numpy(), '\nX_opt =\n',Sigmoid_X(X_opt).numpy())
    print('Maximizing acquisition function - completed!')

    # Results of the optimization
    # Re-scale X_opt to the [0 - 1] scale
    x_opt = Sigmoid_X(X_opt).numpy()
    # Select point with highest acquisition value
    index_opt = np.argsort(loss.numpy(), axis=0)[0]
    x_new = x_opt[index_opt, :]

    #%% ############## Evaluate new design
    x_new_original = scaler_X.inverse_transform(x_new)
    F_new = Test_functions.Viennet(x_new_original)

    #%% ############## Optimization performance metrics
    ind_0 = HV(ref_point=ref_point)
    initial_HV = ind_0(PF_init)
    ind_i = HV(ref_point=ref_point)
    HV_optimization[iter, :] = ind_i(PF)
    AF_i, UCB_i, penal_dist_x_i  = Composed_acquisition_function_penalized_no_classifier(x_new, GPR, Beta, Slack, Sigmoid_X, X_train_n, eff_length, opt = 0)

    print("Iteration:", iter + 1)
    print("New design:", x_new_original[0])
    print("Initial HV:", initial_HV)
    print("Current HV:", ind_i(PF))
    print("Acquisition function:", -AF_i.numpy()[0, 0])
    print("UCB Neg. augmented_Tchebycheff:", UCB_i.numpy()[0, 0])

    # %% ############## Visualization
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    plt.figure(figsize=(4, 7))
    ax1 = plt.subplot(211, projection='3d')
    ax1.scatter(F_train[:n_init, 0], F_train[:n_init, 1], F_train[:n_init, 2], c='green', s=30, alpha=0.4)
    ax1.scatter(F_train[n_init:, 0], F_train[n_init:, 1], F_train[n_init:, 2], c='gray', s=30, alpha=0.4)
    ax1.scatter(PF_init[:, 0], PF_init[:, 1], PF_init[:, 2], marker='s', edgecolors='blue', s=40,
                facecolors="None", lw=1, alpha=0.4)
    ax1.scatter(PF[:, 0], PF[:, 1], PF[:, 2], c='black', s=30)
    ax1.scatter(F_new[:, 0], F_new[:, 1], F_new[:, 2], c='red', s=30, alpha=0.8)
    ax1.set_xlabel('f1')
    ax1.set_ylabel('f2')
    ax1.set_zlabel('f3')
    plt.tight_layout()

    ax2 = plt.subplot(212)
    ax2.plot(HV_optimization[:iter + 1, 0])
    ax2.set_xlabel('Iteration')
    ax2.title.set_text('DHV at iteration ' + str(iter + 1))
    plt.tight_layout()
    plt.show()

    # %% Update training data
    F_train = np.concatenate((F_train, F_new), 0)
    X_train = np.concatenate((X_train, x_new_original), 0)
print('Optimization completed ...')

#%% ######################### DTLZ1 - 6dv ####################################################
#################### Optimization settings (These section can be changed by the designer)
####################################################
test_index = 0 # select a test index from 0 to 4 (initial sampling plan)
total_iter = 190 # Total number of iterations
n_dimensions = 6
num_steps_optimizer = 400  # Number of iterations of SGD-based optimizer (300 - 400)
learning_rate = 0.25  # Learning rate optimizer
n_samples_opt = np.min((25 * n_dimensions, 500)) # Number of points for multi-point optimization of the AF
eff_length = ((0.05 ** 2) * n_dimensions) ** 0.5  # Effective lenght for penalization of similar designs
ref_point = np.array([400., 400., 400.]) # Reference point for calculation of dominated hypervolume
X_bounds = np.concatenate((np.zeros((1,n_dimensions)),np.ones((1,n_dimensions))),0) # Box constraints of the optimization problem
# Settings sr-MOBO approach
rho_tb = 0.65
lambda_tb = 0.01
Beta = 0.95
Slack = 10
# Update weighting vector according to the number of iterations
weights_dic = scipy.io.loadmat('data/DOE_weights_3_obj_190.mat')
weights_sde = weights_dic.get('w_DOE_samples')

########################################################################################################
################## Optimization approach (Do not change if you want to preserve the methodology presented in the JMD publication)
####################################################
# Import initial sampling plan
X_train_dic = scipy.io.loadmat('data/three_objective_6dv.mat')
X_train_lhs = X_train_dic.get('X_initial')

# Evaluate sampling plan
X_train = X_train_lhs[:, :, test_index]
n_init = X_train.shape[0]
Test_function = get_problem("dtlz1", n_var = n_dimensions, n_obj = 3)
F_train = Test_function.evaluate(X_train, return_values_of = ["F"])

# Arrays to track behaviour of optimization approach
HV_optimization = np.zeros((total_iter, 1))
NaN_flag = np.zeros((total_iter, 1))

# Normalization of inputs outside optimization loop as we know its maximum and minimum values
scaler_X = MinMaxScaler()
scaler_X.fit(X_bounds)

# Bayesian optimization loop
for iter in range(total_iter):
    # Sample weights
    weights_tb = weights_sde[iter]
    #%% ############# Find Pareto front and Pareto set
    # Pareto designs: O if dominated, 1 if not dominated
    Pareto_index = Pareto_front(F_train)
    # Pareto front
    PF = F_train[Pareto_index]
    # Pareto designs
    PD = X_train[Pareto_index, :]
    # Create scaler instances
    scaler_Y = MinMaxScaler()
    scaler_Y.fit(F_train)
    # Normalize training data
    F_train_n, X_train_n = scaler_Y.transform(F_train), scaler_X.transform(X_train)

    # Record initial Pareto front and Pareto set
    if iter == 0:
        # Pareto designs
        PF_init = PF
        PD_init = PD

    # Transform data according to Augmented_Tchebycheff function
    F_aug_Tcheb = Augmented_Tchebycheff(X_train_n, F_train_n, weights_tb, rho_tb, lambda_tb)
    F_aug_Tcheb_neg = -F_aug_Tcheb

    #%% ############## GPR model of the negative augmented_Tchebycheff function
    # Build GPR model
    GPR = build_GPR_model(X_train_n, F_aug_Tcheb_neg)

    # Train GPR model
    print('Training GPR model of Augmented Tchebycheff function ...')
    opt1 = gpf.optimizers.Scipy()
    opt1.minimize(GPR.training_loss, variables=GPR.trainable_variables, options=dict(maxiter=1000),
                  method="L-BFGS-B")
    # Check for nan values; if they occur reset model and use adagrad to train GPs
    if np.isnan(GPR.kernel.kernels[0].lengthscales.numpy()).any():
        GPR = build_GPR_model(X_train_n, F_aug_Tcheb_neg)
        optimizer_1 = tf.keras.optimizers.Adagrad(learning_rate=0.01)
        print('Training GPR model using adagrad')

        @tf.function
        def step_1(i):
            optimizer_1.minimize(GPR.training_loss, GPR.trainable_variables)

        for i in tf.range(10000):
            step_1(i)
        NaN_flag[iter, 0] = 1
    print('Training GPR model of Augmented Tchebycheff function completed')

    #%% ############## Optimization of acquisition function
    # Optimization using AD
    # Transformation of variables to enforce box constraints
    lb_X = np.float64(np.zeros(n_dimensions))
    ub_X = np.float64(np.ones(n_dimensions))
    bounds = [lb_X, ub_X]
    Sigmoid_X = tfp.bijectors.Sigmoid(low=lb_X, high=ub_X)

    # Initialization of search points (multi-point search)
    sampler = qmc.LatinHypercube(d=n_dimensions)
    X_initial = sampler.random(n=n_samples_opt)
    # Initial points
    X_opt = tf.Variable(Sigmoid_X.inverse(X_initial), trainable=True, dtype=tf.float64)

    @tf.function
    def get_loss_and_grads():
        with tf.GradientTape() as tape:
            # Specify the variables we want to track for the following operations
            tape.watch(X_opt)
            # Loss function
            loss = Composed_acquisition_function_penalized_no_classifier(X_opt, GPR, Beta, Slack, Sigmoid_X, X_train_n, eff_length, opt=1)[0]
            # Compute the gradient of the loss wrt the trainable_variables
            grads = tape.gradient(loss, X_opt)
        return loss, grads

    # Optimizer for the AF
    optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)

    print('Maximizing acquisition function ...')
    for i in range(num_steps_optimizer):
        # Compute loss and gradients
        loss, grads = get_loss_and_grads()
        # Update the training variables
        optimizer.apply_gradients(zip([grads], [X_opt]))
        # print('l =\n',-loss.numpy(), '\ng =\n', grads.numpy(), '\nX_opt =\n',Sigmoid_X(X_opt).numpy())
    print('Maximizing acquisition function - completed!')

    # Results of the optimization
    # Re-scale X_opt to the [0 - 1] scale
    x_opt = Sigmoid_X(X_opt).numpy()
    # Select point with highest acquisition value
    index_opt = np.argsort(loss.numpy(), axis=0)[0]
    x_new = x_opt[index_opt, :]

    #%% ############## Evaluate new design
    x_new_original = scaler_X.inverse_transform(x_new)
    F_new = Test_function.evaluate(x_new_original, return_values_of = ["F"])

    #%% ############## Optimization performance metrics
    ind_0 = HV(ref_point=ref_point)
    initial_HV = ind_0(PF_init)
    ind_i = HV(ref_point=ref_point)
    HV_optimization[iter, :] = ind_i(PF)
    AF_i, UCB_i, penal_dist_x_i  = Composed_acquisition_function_penalized_no_classifier(x_new, GPR, Beta, Slack, Sigmoid_X, X_train_n, eff_length, opt = 0)

    print("Iteration:", iter + 1)
    print("New design:", x_new_original[0])
    print("Initial HV:", initial_HV)
    print("Current HV:", ind_i(PF))
    print("Acquisition function:", -AF_i.numpy()[0, 0])
    print("UCB Neg. augmented_Tchebycheff:", UCB_i.numpy()[0, 0])

    # %% ############## Visualization
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    plt.figure(figsize=(4, 7))
    ax1 = plt.subplot(211, projection='3d')
    ax1.scatter(F_train[:n_init, 0], F_train[:n_init, 1], F_train[:n_init, 2], c='green', s=30, alpha=0.4)
    ax1.scatter(F_train[n_init:, 0], F_train[n_init:, 1], F_train[n_init:, 2], c='gray', s=30, alpha=0.4)
    ax1.scatter(PF_init[:, 0], PF_init[:, 1], PF_init[:, 2], marker='s', edgecolors='blue', s=40,
                facecolors="None", lw=1, alpha=0.4)
    ax1.scatter(PF[:, 0], PF[:, 1], PF[:, 2], c='black', s=30)
    ax1.scatter(F_new[:, 0], F_new[:, 1], F_new[:, 2], c='red', s=30, alpha=0.8)
    ax1.set_xlabel('f1')
    ax1.set_ylabel('f2')
    ax1.set_zlabel('f3')
    plt.tight_layout()

    ax2 = plt.subplot(212)
    ax2.plot(HV_optimization[:iter + 1, 0])
    ax2.set_xlabel('Iteration')
    ax2.title.set_text('DHV at iteration ' + str(iter + 1))
    plt.tight_layout()
    plt.show()

    # %% Update training data
    F_train = np.concatenate((F_train, F_new), 0)
    X_train = np.concatenate((X_train, x_new_original), 0)
print('Optimization completed ...')

#%% ######################### DTLZ2 - 6dv ####################################################
#################### Optimization settings (These section can be changed by the designer)
####################################################
test_index = 0 # select a test index from 0 to 4 (initial sampling plan)
total_iter = 190 # Total number of iterations
n_dimensions = 6
num_steps_optimizer = 400  # Number of iterations of SGD-based optimizer (300 - 400)
learning_rate = 0.25  # Learning rate optimizer
n_samples_opt = np.min((25 * n_dimensions, 500)) # Number of points for multi-point optimization of the AF
eff_length = ((0.05 ** 2) * n_dimensions) ** 0.5  # Effective lenght for penalization of similar designs
ref_point = np.array([2.5, 2.5, 2.5]) # Reference point for calculation of dominated hypervolume
X_bounds = np.concatenate((np.zeros((1,n_dimensions)),np.ones((1,n_dimensions))),0) # Box constraints of the optimization problem
# Settings sr-MOBO approach
rho_tb = 0.65
lambda_tb = 0.01
Beta = 0.95
Slack = 10
# Update weighting vector according to the number of iterations
weights_dic = scipy.io.loadmat('data/DOE_weights_3_obj_190.mat')
weights_sde = weights_dic.get('w_DOE_samples')

########################################################################################################
################## Optimization approach (Do not change if you want to preserve the methodology presented in the JMD publication)
####################################################
# Import initial sampling plan
X_train_dic = scipy.io.loadmat('data/three_objective_6dv.mat')
X_train_lhs = X_train_dic.get('X_initial')

# Evaluate sampling plan
X_train = X_train_lhs[:, :, test_index]
n_init = X_train.shape[0]
Test_function = get_problem("dtlz2", n_var = n_dimensions, n_obj = 3)
F_train = Test_function.evaluate(X_train, return_values_of = ["F"])

# Arrays to track behaviour of optimization approach
HV_optimization = np.zeros((total_iter, 1))
NaN_flag = np.zeros((total_iter, 2))

# Normalization of inputs outside optimization loop as we know its maximum and minimum values
scaler_X = MinMaxScaler()
scaler_X.fit(X_bounds)

# Bayesian optimization loop
for iter in range(total_iter):
    # Sample weights
    weights_tb = weights_sde[iter]
    #%% ############# Find Pareto front and Pareto set
    # Pareto designs: O if dominated, 1 if not dominated
    Pareto_index = Pareto_front(F_train)
    # Pareto front
    PF = F_train[Pareto_index]
    # Pareto designs
    PD = X_train[Pareto_index, :]
    # Create scaler instances
    scaler_Y = MinMaxScaler()
    scaler_Y.fit(F_train)
    # Normalize training data
    F_train_n, X_train_n = scaler_Y.transform(F_train), scaler_X.transform(X_train)

    # Record initial Pareto front and Pareto set
    if iter == 0:
        # Pareto designs
        PF_init = PF
        PD_init = PD

    # Transform data according to Augmented_Tchebycheff function
    F_aug_Tcheb = Augmented_Tchebycheff(X_train_n, F_train_n, weights_tb, rho_tb, lambda_tb)
    F_aug_Tcheb_neg = -F_aug_Tcheb

    #%% ############## GPR model of the negative augmented_Tchebycheff function
    # Build GPR model
    GPR = build_GPR_model(X_train_n, F_aug_Tcheb_neg)

    # Train GPR model
    print('Training GPR model of Augmented Tchebycheff function ...')
    opt1 = gpf.optimizers.Scipy()
    opt1.minimize(GPR.training_loss, variables=GPR.trainable_variables, options=dict(maxiter=1000),
                  method="L-BFGS-B")
    # Check for nan values; if they occur reset model and use adagrad to train GPs
    if np.isnan(GPR.kernel.kernels[0].lengthscales.numpy()).any():
        GPR = build_GPR_model(X_train_n, F_aug_Tcheb_neg)
        optimizer_1 = tf.keras.optimizers.Adagrad(learning_rate=0.01)
        print('Training GPR model using adagrad')

        @tf.function
        def step_1(i):
            optimizer_1.minimize(GPR.training_loss, GPR.trainable_variables)

        for i in tf.range(10000):
            step_1(i)
        NaN_flag[iter, 0] = 1
    print('Training GPR model of Augmented Tchebycheff function completed')

    #%% ############## Optimization of acquisition function
    # Optimization using AD
    # Transformation of variables to enforce box constraints
    lb_X = np.float64(np.zeros(n_dimensions))
    ub_X = np.float64(np.ones(n_dimensions))
    bounds = [lb_X, ub_X]
    Sigmoid_X = tfp.bijectors.Sigmoid(low=lb_X, high=ub_X)

    # Initialization of search points (multi-point search)
    sampler = qmc.LatinHypercube(d=n_dimensions)
    X_initial = sampler.random(n=n_samples_opt)
    # Initial points
    X_opt = tf.Variable(Sigmoid_X.inverse(X_initial), trainable=True, dtype=tf.float64)

    @tf.function
    def get_loss_and_grads():
        with tf.GradientTape() as tape:
            # Specify the variables we want to track for the following operations
            tape.watch(X_opt)
            # Loss function
            loss = Composed_acquisition_function_penalized_no_classifier(X_opt, GPR, Beta, Slack, Sigmoid_X, X_train_n, eff_length, opt=1)[0]
            # Compute the gradient of the loss wrt the trainable_variables
            grads = tape.gradient(loss, X_opt)
        return loss, grads

    # Optimizer for the AF
    optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate)

    print('Maximizing acquisition function ...')
    for i in range(num_steps_optimizer):
        # Compute loss and gradients
        loss, grads = get_loss_and_grads()
        # Update the training variables
        optimizer.apply_gradients(zip([grads], [X_opt]))
        # print('l =\n',-loss.numpy(), '\ng =\n', grads.numpy(), '\nX_opt =\n',Sigmoid_X(X_opt).numpy())
    print('Maximizing acquisition function - completed!')

    # Results of the optimization
    # Re-scale X_opt to the [0 - 1] scale
    x_opt = Sigmoid_X(X_opt).numpy()
    # Select point with highest acquisition value
    index_opt = np.argsort(loss.numpy(), axis=0)[0]
    x_new = x_opt[index_opt, :]

    #%% ############## Evaluate new design
    x_new_original = scaler_X.inverse_transform(x_new)
    F_new = Test_function.evaluate(x_new_original, return_values_of = ["F"])

    #%% ############## Optimization performance metrics
    ind_0 = HV(ref_point=ref_point)
    initial_HV = ind_0(PF_init)
    ind_i = HV(ref_point=ref_point)
    HV_optimization[iter, :] = ind_i(PF)
    AF_i, UCB_i, penal_dist_x_i  = Composed_acquisition_function_penalized_no_classifier(x_new, GPR, Beta, Slack, Sigmoid_X, X_train_n, eff_length, opt = 0)

    print("Iteration:", iter + 1)
    print("New design:", x_new_original[0])
    print("Initial HV:", initial_HV)
    print("Current HV:", ind_i(PF))
    print("Acquisition function:", -AF_i.numpy()[0, 0])
    print("UCB Neg. augmented_Tchebycheff:", UCB_i.numpy()[0, 0])

    # %% ############## Visualization
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

    plt.figure(figsize=(4, 7))
    ax1 = plt.subplot(211, projection='3d')
    ax1.scatter(F_train[:n_init, 0], F_train[:n_init, 1], F_train[:n_init, 2], c='green', s=30, alpha=0.4)
    ax1.scatter(F_train[n_init:, 0], F_train[n_init:, 1], F_train[n_init:, 2], c='gray', s=30, alpha=0.4)
    ax1.scatter(PF_init[:, 0], PF_init[:, 1], PF_init[:, 2], marker='s', edgecolors='blue', s=40,
                facecolors="None", lw=1, alpha=0.4)
    ax1.scatter(PF[:, 0], PF[:, 1], PF[:, 2], c='black', s=30)
    ax1.scatter(F_new[:, 0], F_new[:, 1], F_new[:, 2], c='red', s=30, alpha=0.8)
    ax1.set_xlabel('f1')
    ax1.set_ylabel('f2')
    ax1.set_zlabel('f3')
    plt.tight_layout()

    ax2 = plt.subplot(212)
    ax2.plot(HV_optimization[:iter + 1, 0])
    ax2.set_xlabel('Iteration')
    ax2.title.set_text('DHV at iteration ' + str(iter + 1))
    plt.tight_layout()
    plt.show()

    # %% Update training data
    F_train = np.concatenate((F_train, F_new), 0)
    X_train = np.concatenate((X_train, x_new_original), 0)
print('Optimization completed ...')