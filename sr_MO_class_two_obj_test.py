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

# Import in-house modules for multiobjective optimization
import Test_functions
from optimization_files.sr_MO_BO_3GP_approach import Augmented_Tchebycheff, Composed_acquisition_function_penalized
from optimization_files.vanilla_GP_model import build_GPR_model, build_GPC_model
from optimization_files.find_pareto import Pareto_front

#%% ######################### Fonseca - 2dv ####################################################
#################### Optimization settings (These section can be changed by the designer)
####################################################
test_index = 0 # select a test index from 0 to 4 (initial sampling plan)
total_iter = 100 # Total number of iterations
n_dimensions = 2
num_steps_optimizer = 400  # Number of iterations of SGD-based optimizer (300 - 400)
learning_rate = 0.25  # Learning rate optimizer
n_samples_opt = np.min((25 * n_dimensions, 500)) # Number of points for multi-point optimization of the AF
eff_length = ((0.05 ** 2) * n_dimensions) ** 0.5  # Effective lenght for penalization of similar designs
ref_point = np.array([1., 1.]) # Reference point for calculation of dominated hypervolume
X_bounds = np.concatenate((-4*np.ones((1,n_dimensions)),4*np.ones((1,n_dimensions))),0) # Box constraints of the optimization problem
# Settings sr-MOBO approach
rho_tb = 0.65
lambda_tb = 0.01
Beta = 0.95
Slack = 10
# Update weighting vector according to the number of iterations
weights_dic = scipy.io.loadmat('data/DOE_weights_2_obj_100.mat')
weights_sde = weights_dic.get('w_DOE_samples')

########################################################################################################
################## Optimization approach (Do not change if you want to preserve the methodology presented in the JMD publication)
####################################################
# Import initial sampling plan
X_train_dic = scipy.io.loadmat('data/two_objective_Fonseca_2_dv.mat')
X_train_lhs = X_train_dic.get('X_initial')

# Evaluate sampling plan
X_train = X_train_lhs[:, :, test_index]
n_init = X_train.shape[0]
F_train = Test_functions.Fonseca(X_train, n_dimensions)

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
    # Normalize PF, PD, and training data
    PFn, PDn = scaler_Y.transform(PF), scaler_X.transform(PD)
    F_train_n, X_train_n = scaler_Y.transform(F_train), scaler_X.transform(X_train)

    # Record initial Pareto front and Pareto set
    if iter == 0:
        # Pareto designs
        PF_init = PF
        PD_init = PD
    # Transform initial design according to current transformation
    PDn_init = scaler_X.transform(PD_init)
    PFn_init = scaler_Y.transform(PF_init)

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

    #%% ############## GP classifier model of the negative augmented_Tchebycheff function
    # Build GPC model
    fc_train = (Pareto_index.reshape(-1, 1).astype(float))
    GPC = build_GPC_model(X_train_n, fc_train)

    # %% Train GP classifier
    # Train GP classifier
    optc = gpf.optimizers.Scipy()
    print('Training GPC model ...')
    optc.minimize(GPC.training_loss, variables=GPC.trainable_variables, options=dict(maxiter=1000),
                  method="L-BFGS-B")
    # Check for nan values; if they occur reset model and use adagrad to train GPs
    if np.isnan(GPC.kernel.kernels[0].lengthscales.numpy()).any():
        GPC = build_GPC_model(X_train_n, fc_train)
        optimizer_2 = tf.keras.optimizers.Adagrad(learning_rate=0.01)
        print('Training GPC model using adagrad')

        @tf.function
        def step_2(i):
            optimizer_2.minimize(GPC.training_loss, GPC.trainable_variables)

        for i in tf.range(10000):
            step_2(i)
        NaN_flag[iter, 1] = 1
    print('Training GPC model completed')

    #%% ############## Best current observation for GPC improvement
    y_c_best = GPC.predict_y(PDn)[0].numpy().mean()

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
            loss = Composed_acquisition_function_penalized(X_opt, GPR, GPC, Beta, Slack, y_c_best, Sigmoid_X, X_train_n, eff_length, opt=1)[0]
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
    F_new = Test_functions.Fonseca(x_new_original, n_dimensions)

    #%% ############## Optimization performance metrics
    ind_0 = HV(ref_point=ref_point)
    initial_HV = ind_0(PF_init)
    ind_i = HV(ref_point=ref_point)
    HV_optimization[iter, :] = ind_i(PF)
    AF_i, UCB_i, EI_i, penal_dist_x_i  = Composed_acquisition_function_penalized(x_new,GPR, GPC, Beta, Slack, y_c_best, Sigmoid_X, X_train_n, eff_length, opt = 0)

    print("Iteration:", iter + 1)
    print("New design:", x_new_original[0])
    print("Initial HV:", initial_HV)
    print("Current HV:", ind_i(PF))
    print("Acquisition function:", -AF_i.numpy()[0, 0])
    print("UCB Neg. augmented_Tchebycheff:", UCB_i.numpy()[0, 0])
    print("EI classifier:", EI_i.numpy()[0, 0])

    #%% ############## Visualization
    plt.figure(figsize=(4, 7))
    ax1 = plt.subplot(211)
    ax1.grid()
    plt.scatter(F_train[:n_init, 0], F_train[:n_init, 1], c='green', alpha=0.5)
    plt.scatter(F_train[n_init:, 0], F_train[n_init:, 1], c='gray', alpha=0.3)
    plt.scatter(PF[:, 0], PF[:, 1], c='black')
    plt.scatter(PF_init[:, 0], PF_init[:, 1], c='blue')
    plt.plot(F_new[0:, 0], F_new[0:, 1], 'ro', alpha=0.8)
    ax1.set_xlabel('$f_1$')
    ax1.set_ylabel('$f_2$')
    ax1.set_title('Iteration ' + str(iter + 1))
    plt.tight_layout()

    ax2 = plt.subplot(212)
    ax2.plot(HV_optimization[:iter+1,0])
    ax2.set_xlabel('Iteration')
    ax2.title.set_text('DHV at iteration ' + str(iter + 1))
    plt.tight_layout()
    plt.show()

    # %% Update training data
    F_train = np.concatenate((F_train, F_new), 0)
    X_train = np.concatenate((X_train, x_new_original), 0)
print('Optimization completed ...')

#%% ######################### Kursawe - 3dv ####################################################
#################### Optimization settings (These section can be changed by the designer)
####################################################
test_index = 0 # select a test index from 0 to 4 (initial sampling plan)
total_iter = 100 # Total number of iterations
n_dimensions = 3
num_steps_optimizer = 400  # Number of iterations of SGD-based optimizer (300 - 400)
learning_rate = 0.25  # Learning rate optimizer
n_samples_opt = np.min((25 * n_dimensions, 500)) # Number of points for multi-point optimization of the AF
eff_length = ((0.05 ** 2) * n_dimensions) ** 0.5  # Effective lenght for penalization of similar designs
ref_point = np.array([-6., 15.])# Reference point for calculation of dominated hypervolume
X_bounds = np.concatenate((-5*np.ones((1,n_dimensions)),5*np.ones((1,n_dimensions))),0) # Box constraints of the optimization problem
# Settings sr-MOBO approach
rho_tb = 0.65
lambda_tb = 0.01
Beta = 0.95
Slack = 10
# Update weighting vector according to the number of iterations
weights_dic = scipy.io.loadmat('data/DOE_weights_2_obj_100.mat')
weights_sde = weights_dic.get('w_DOE_samples')


########################################################################################################
################## Optimization approach (Do not change if you want to preserve the methodology presented in the JMD publication)
####################################################
# Import initial sampling plan
X_train_dic = scipy.io.loadmat('data/two_objective_Kursawe_3_dv.mat')
X_train_lhs = X_train_dic.get('X_initial')

# Evaluate sampling plan
X_train = X_train_lhs[:, :, test_index]
n_init = X_train.shape[0]
F_train = Test_functions.Kursawe(X_train)

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
    # Normalize PF, PD, and training data
    PFn, PDn = scaler_Y.transform(PF), scaler_X.transform(PD)
    F_train_n, X_train_n = scaler_Y.transform(F_train), scaler_X.transform(X_train)

    # Record initial Pareto front and Pareto set
    if iter == 0:
        # Pareto designs
        PF_init = PF
        PD_init = PD
    # Transform initial design according to current transformation
    PDn_init = scaler_X.transform(PD_init)
    PFn_init = scaler_Y.transform(PF_init)

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

    #%% ############## GP classifier model of the negative augmented_Tchebycheff function
    # Build GPC model
    fc_train = (Pareto_index.reshape(-1, 1).astype(float))
    GPC = build_GPC_model(X_train_n, fc_train)

    # %% Train GP classifier
    # Train GP classifier
    optc = gpf.optimizers.Scipy()
    print('Training GPC model ...')
    optc.minimize(GPC.training_loss, variables=GPC.trainable_variables, options=dict(maxiter=1000),
                  method="L-BFGS-B")
    # Check for nan values; if they occur reset model and use adagrad to train GPs
    if np.isnan(GPC.kernel.kernels[0].lengthscales.numpy()).any():
        GPC = build_GPC_model(X_train_n, fc_train)
        optimizer_2 = tf.keras.optimizers.Adagrad(learning_rate=0.01)
        print('Training GPC model using adagrad')

        @tf.function
        def step_2(i):
            optimizer_2.minimize(GPC.training_loss, GPC.trainable_variables)

        for i in tf.range(10000):
            step_2(i)
        NaN_flag[iter, 1] = 1
    print('Training GPC model completed')

    #%% ############## Best current observation for GPC improvement
    y_c_best = GPC.predict_y(PDn)[0].numpy().mean()

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
            loss = Composed_acquisition_function_penalized(X_opt, GPR, GPC, Beta, Slack, y_c_best, Sigmoid_X, X_train_n, eff_length, opt=1)[0]
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
    F_new = Test_functions.Kursawe(x_new_original)

    #%% ############## Optimization performance metrics
    ind_0 = HV(ref_point=ref_point)
    initial_HV = ind_0(PF_init)
    ind_i = HV(ref_point=ref_point)
    HV_optimization[iter, :] = ind_i(PF)
    AF_i, UCB_i, EI_i, penal_dist_x_i  = Composed_acquisition_function_penalized(x_new,GPR, GPC, Beta, Slack, y_c_best, Sigmoid_X, X_train_n, eff_length, opt = 0)

    print("Iteration:", iter + 1)
    print("New design:", x_new_original[0])
    print("Initial HV:", initial_HV)
    print("Current HV:", ind_i(PF))
    print("Acquisition function:", -AF_i.numpy()[0, 0])
    print("UCB Neg. augmented_Tchebycheff:", UCB_i.numpy()[0, 0])
    print("EI classifier:", EI_i.numpy()[0, 0])

    #%% ############## Visualization
    plt.figure(figsize=(4, 7))
    ax1 = plt.subplot(211)
    ax1.grid()
    plt.scatter(F_train[:n_init, 0], F_train[:n_init, 1], c='green', alpha=0.5)
    plt.scatter(F_train[n_init:, 0], F_train[n_init:, 1], c='gray', alpha=0.3)
    plt.scatter(PF[:, 0], PF[:, 1], c='black')
    plt.scatter(PF_init[:, 0], PF_init[:, 1], c='blue')
    plt.plot(F_new[0:, 0], F_new[0:, 1], 'ro', alpha=0.8)
    ax1.set_xlabel('$f_1$')
    ax1.set_ylabel('$f_2$')
    ax1.set_title('Iteration ' + str(iter + 1))
    plt.tight_layout()

    ax2 = plt.subplot(212)
    ax2.plot(HV_optimization[:iter+1,0])
    ax2.set_xlabel('Iteration')
    ax2.title.set_text('DHV at iteration ' + str(iter + 1))
    plt.tight_layout()
    plt.show()

    # %% Update training data
    F_train = np.concatenate((F_train, F_new), 0)
    X_train = np.concatenate((X_train, x_new_original), 0)
print('Optimization completed ...')


#%% ######################### Omni - 6dv ####################################################
#################### Optimization settings (These section can be changed by the designer)
####################################################
test_index = 1 # select a test index from 0 to 4 (initial sampling plan)
total_iter = 200 # Total number of iterations
n_dimensions = 6
num_steps_optimizer = 400  # Number of iterations of SGD-based optimizer (300 - 400)
learning_rate = 0.25  # Learning rate optimizer
n_samples_opt = np.min((25 * n_dimensions, 500)) # Number of points for multi-point optimization of the AF
eff_length = ((0.05 ** 2) * n_dimensions) ** 0.5  # Effective lenght for penalization of similar designs
ref_point = np.array([0., 0.]) # Reference point for calculation of dominated hypervolume
X_bounds = np.concatenate((np.zeros((1,n_dimensions)),6*np.ones((1,n_dimensions))),0) # Box constraints of the optimization problem
# Settings sr-MOBO approach
rho_tb = 0.65
lambda_tb = 0.01
Beta = 0.95
Slack = 10
# Update weighting vector according to the number of iterations
weights_dic = scipy.io.loadmat('data/DOE_weights_2_obj_200.mat')
weights_sde = weights_dic.get('w_DOE_samples')

########################################################################################################
################## Optimization approach (Do not change if you want to preserve the methodology presented in the JMD publication)
####################################################
# Import OmniTest from pymoo
from pymoo.problems.multi.omnitest import OmniTest
Test_function = OmniTest(n_var = n_dimensions)

# Import initial sampling plan
X_train_dic = scipy.io.loadmat('data/two_objective_Omni_6_dv.mat')
X_train_lhs = X_train_dic.get('X_initial')

# Evaluate sampling plan
X_train = X_train_lhs[:, :, test_index]
n_init = X_train.shape[0]
F_train = Test_function.evaluate(X_train, return_values_of=["F"])

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
    # Normalize PF, PD, and training data
    PFn, PDn = scaler_Y.transform(PF), scaler_X.transform(PD)
    F_train_n, X_train_n = scaler_Y.transform(F_train), scaler_X.transform(X_train)

    # Record initial Pareto front and Pareto set
    if iter == 0:
        # Pareto designs
        PF_init = PF
        PD_init = PD
    # Transform initial design according to current transformation
    PDn_init = scaler_X.transform(PD_init)
    PFn_init = scaler_Y.transform(PF_init)

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

    #%% ############## GP classifier model of the negative augmented_Tchebycheff function
    # Build GPC model
    fc_train = (Pareto_index.reshape(-1, 1).astype(float))
    GPC = build_GPC_model(X_train_n, fc_train)

    # %% Train GP classifier
    # Train GP classifier
    optc = gpf.optimizers.Scipy()
    print('Training GPC model ...')
    optc.minimize(GPC.training_loss, variables=GPC.trainable_variables, options=dict(maxiter=1000),
                  method="L-BFGS-B")
    # Check for nan values; if they occur reset model and use adagrad to train GPs
    if np.isnan(GPC.kernel.kernels[0].lengthscales.numpy()).any():
        GPC = build_GPC_model(X_train_n, fc_train)
        optimizer_2 = tf.keras.optimizers.Adagrad(learning_rate=0.01)
        print('Training GPC model using adagrad')

        @tf.function
        def step_2(i):
            optimizer_2.minimize(GPC.training_loss, GPC.trainable_variables)

        for i in tf.range(10000):
            step_2(i)
        NaN_flag[iter, 1] = 1
    print('Training GPC model completed')

    #%% ############## Best current observation for GPC improvement
    y_c_best = GPC.predict_y(PDn)[0].numpy().mean()

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
            loss = Composed_acquisition_function_penalized(X_opt, GPR, GPC, Beta, Slack, y_c_best, Sigmoid_X, X_train_n, eff_length, opt=1)[0]
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
    F_new = Test_function.evaluate(x_new_original, return_values_of=["F"])

    #%% ############## Optimization performance metrics
    ind_0 = HV(ref_point=ref_point)
    initial_HV = ind_0(PF_init)
    ind_i = HV(ref_point=ref_point)
    HV_optimization[iter, :] = ind_i(PF)
    AF_i, UCB_i, EI_i, penal_dist_x_i  = Composed_acquisition_function_penalized(x_new,GPR, GPC, Beta, Slack, y_c_best, Sigmoid_X, X_train_n, eff_length, opt = 0)

    print("Iteration:", iter + 1)
    print("New design:", x_new_original[0])
    print("Initial HV:", initial_HV)
    print("Current HV:", ind_i(PF))
    print("Acquisition function:", -AF_i.numpy()[0, 0])
    print("UCB Neg. augmented_Tchebycheff:", UCB_i.numpy()[0, 0])
    print("EI classifier:", EI_i.numpy()[0, 0])

    #%% ############## Visualization
    plt.figure(figsize=(4, 7))
    ax1 = plt.subplot(211)
    ax1.grid()
    plt.scatter(F_train[:n_init, 0], F_train[:n_init, 1], c='green', alpha=0.5)
    plt.scatter(F_train[n_init:, 0], F_train[n_init:, 1], c='gray', alpha=0.3)
    plt.scatter(PF[:, 0], PF[:, 1], c='black')
    plt.scatter(PF_init[:, 0], PF_init[:, 1], c='blue')
    plt.plot(F_new[0:, 0], F_new[0:, 1], 'ro', alpha=0.8)
    ax1.set_xlabel('$f_1$')
    ax1.set_ylabel('$f_2$')
    ax1.set_title('Iteration ' + str(iter + 1))
    plt.tight_layout()

    ax2 = plt.subplot(212)
    ax2.plot(HV_optimization[:iter+1,0])
    ax2.set_xlabel('Iteration')
    ax2.title.set_text('DHV at iteration ' + str(iter + 1))
    plt.tight_layout()
    plt.show()

    # %% Update training data
    F_train = np.concatenate((F_train, F_new), 0)
    X_train = np.concatenate((X_train, x_new_original), 0)
print('Optimization completed ...')


#%% ######################### ZDT3 - 6dv ####################################################
#################### Optimization settings (These section can be changed by the designer)
####################################################
test_index = 0 # select a test index from 0 to 4 (initial sampling plan)
total_iter = 200 # Total number of iterations
n_dimensions = 6
num_steps_optimizer = 400  # Number of iterations of SGD-based optimizer (300 - 400)
learning_rate = 0.25  # Learning rate optimizer
n_samples_opt = np.min((25 * n_dimensions, 500)) # Number of points for multi-point optimization of the AF
eff_length = ((0.05 ** 2) * n_dimensions) ** 0.5  # Effective lenght for penalization of similar designs
ref_point = np.array([1., 7.]) # Reference point for calculation of dominated hypervolume
X_bounds = np.concatenate((np.zeros((1,n_dimensions)),np.ones((1,n_dimensions))),0) # Box constraints of the optimization problem
# Settings sr-MOBO approach
rho_tb = 0.65
lambda_tb = 0.01
Beta = 0.95
Slack = 10
# Update weighting vector according to the number of iterations
weights_dic = scipy.io.loadmat('data/DOE_weights_2_obj_200.mat')
weights_sde = weights_dic.get('w_DOE_samples')

########################################################################################################
################## Optimization approach (Do not change if you want to preserve the methodology presented in the JMD publication)
####################################################
# Import initial sampling plan
X_train_dic = scipy.io.loadmat('data/two_objective_ZDT3_6_dv.mat')
X_train_lhs = X_train_dic.get('X_initial')

# Evaluate sampling plan
X_train = X_train_lhs[:, :, test_index]
n_init = X_train.shape[0]
F_train = Test_functions.ZDT3(X_train, n_dimensions)

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
    # Normalize PF, PD, and training data
    PFn, PDn = scaler_Y.transform(PF), scaler_X.transform(PD)
    F_train_n, X_train_n = scaler_Y.transform(F_train), scaler_X.transform(X_train)

    # Record initial Pareto front and Pareto set
    if iter == 0:
        # Pareto designs
        PF_init = PF
        PD_init = PD
    # Transform initial design according to current transformation
    PDn_init = scaler_X.transform(PD_init)
    PFn_init = scaler_Y.transform(PF_init)

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

    #%% ############## GP classifier model of the negative augmented_Tchebycheff function
    # Build GPC model
    fc_train = (Pareto_index.reshape(-1, 1).astype(float))
    GPC = build_GPC_model(X_train_n, fc_train)

    # %% Train GP classifier
    # Train GP classifier
    optc = gpf.optimizers.Scipy()
    print('Training GPC model ...')
    optc.minimize(GPC.training_loss, variables=GPC.trainable_variables, options=dict(maxiter=1000),
                  method="L-BFGS-B")
    # Check for nan values; if they occur reset model and use adagrad to train GPs
    if np.isnan(GPC.kernel.kernels[0].lengthscales.numpy()).any():
        GPC = build_GPC_model(X_train_n, fc_train)
        optimizer_2 = tf.keras.optimizers.Adagrad(learning_rate=0.01)
        print('Training GPC model using adagrad')

        @tf.function
        def step_2(i):
            optimizer_2.minimize(GPC.training_loss, GPC.trainable_variables)

        for i in tf.range(10000):
            step_2(i)
        NaN_flag[iter, 1] = 1
    print('Training GPC model completed')

    #%% ############## Best current observation for GPC improvement
    y_c_best = GPC.predict_y(PDn)[0].numpy().mean()

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
            loss = Composed_acquisition_function_penalized(X_opt, GPR, GPC, Beta, Slack, y_c_best, Sigmoid_X, X_train_n, eff_length, opt=1)[0]
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
    F_new = Test_functions.ZDT3(x_new_original, n_dimensions)

    #%% ############## Optimization performance metrics
    ind_0 = HV(ref_point=ref_point)
    initial_HV = ind_0(PF_init)
    ind_i = HV(ref_point=ref_point)
    HV_optimization[iter, :] = ind_i(PF)
    AF_i, UCB_i, EI_i, penal_dist_x_i  = Composed_acquisition_function_penalized(x_new,GPR, GPC, Beta, Slack, y_c_best, Sigmoid_X, X_train_n, eff_length, opt = 0)

    print("Iteration:", iter + 1)
    print("New design:", x_new_original[0])
    print("Initial HV:", initial_HV)
    print("Current HV:", ind_i(PF))
    print("Acquisition function:", -AF_i.numpy()[0, 0])
    print("UCB Neg. augmented_Tchebycheff:", UCB_i.numpy()[0, 0])
    print("EI classifier:", EI_i.numpy()[0, 0])

    #%% ############## Visualization
    plt.figure(figsize=(4, 7))
    ax1 = plt.subplot(211)
    ax1.grid()
    plt.scatter(F_train[:n_init, 0], F_train[:n_init, 1], c='green', alpha=0.5)
    plt.scatter(F_train[n_init:, 0], F_train[n_init:, 1], c='gray', alpha=0.3)
    plt.scatter(PF[:, 0], PF[:, 1], c='black')
    plt.scatter(PF_init[:, 0], PF_init[:, 1], c='blue')
    plt.plot(F_new[0:, 0], F_new[0:, 1], 'ro', alpha=0.8)
    ax1.set_xlabel('$f_1$')
    ax1.set_ylabel('$f_2$')
    ax1.set_title('Iteration ' + str(iter + 1))
    plt.tight_layout()

    ax2 = plt.subplot(212)
    ax2.plot(HV_optimization[:iter+1,0])
    ax2.set_xlabel('Iteration')
    ax2.title.set_text('DHV at iteration ' + str(iter + 1))
    plt.tight_layout()
    plt.show()

    # %% Update training data
    F_train = np.concatenate((F_train, F_new), 0)
    X_train = np.concatenate((X_train, x_new_original), 0)
print('Optimization completed ...')
