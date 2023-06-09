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
from optimization_files.Euclidean_EI import Penalized_EEI as EEI
from optimization_files.vanilla_GP_model import build_GPR_model_EPDC
from optimization_files.find_pareto import Pareto_front

#%% ######################### Fonseca - 2dv ####################################################
#################### Optimization settings (These section can be changed by the designer)
####################################################
test_index = 0 # select a test index from 0 to 4 (initial sampling plan)
total_iter = 100 # Total number of iterations
num_steps_optimizer = 400  # Number of iterations of SGD-based optimizer (300 - 400)
learning_rate = 0.25  # Learning rate optimizer
n_dimensions = 2
n_samples_opt = np.min((25 * n_dimensions, 500)) # Number of points for multi-point optimization of the AF
eff_length = ((0.05 ** 2) * n_dimensions) ** 0.5  # Effective lenght for penalization of similar designs
ref_point = np.array([1., 1.]) # Reference point for calculation of dominated hypervolume
X_bounds = np.concatenate((-4*np.ones((1,n_dimensions)),4*np.ones((1,n_dimensions))),0) # Box constraints of the optimization problem

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
n_objectives = F_train.shape[1]  # number of objectives

# Arrays to track behaviour of optimization approach
HV_optimization = np.zeros((total_iter, 1))
NaN_flag = np.zeros((total_iter, 2))

# Normalization of inputs outside optimization loop as we know its maximum and minimum values
scaler_X = MinMaxScaler()
scaler_X.fit(X_bounds)

# Bayesian optimization loop
for iter in range(total_iter):
    #%% ############# Find Pareto front and Pareto set
    # Pareto designs: O if dominated, 1 if not dominated
    Pareto_index = Pareto_front(F_train)
    # Pareto front
    PF = F_train[Pareto_index]
    # Pareto designs
    PD = X_train[Pareto_index, :]
    ################## Sort Pareto front for multi-objective optimization code
    ################## that uses EEI approach (needed!!)
    PF = PF[PF[:, 0].argsort(), :]
    PD = X_train[Pareto_index, :]
    PD = PD[PF[:, 0].argsort(), :]
    ##################
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

    #%% ############## GP models
    # Training data for GPR models
    f1_train, f2_train = np.hsplit(F_train_n, n_objectives)

    # Build GPR models
    m1 = build_GPR_model_EPDC(X_train_n, f1_train)
    m2 = build_GPR_model_EPDC(X_train_n, f2_train)

    # Train GPR model 1
    print('Training GPR model 1 ...')
    opt1 = gpf.optimizers.Scipy()
    opt1.minimize(m1.training_loss, variables=m1.trainable_variables, options=dict(maxiter=1000),
                  method="L-BFGS-B")
    # Check for nan values; if they occur reset model and use adagrad to train GPs
    if np.isnan(m1.kernel.kernels[0].lengthscales.numpy()).any():
        m1 = build_GPR_model_EPDC(X_train_n, f1_train)
        optimizer_1 = tf.keras.optimizers.Adagrad(learning_rate=0.01)
        print('Training GP model 1 using adagrad')

        @tf.function
        def step_1(i):
            optimizer_1.minimize(m1.training_loss, m1.trainable_variables)
        for i in tf.range(10000):
            step_1(i)
        NaN_flag[iter, 0] = 1
    print('Training GPR model 1 completed')

    # Train GPR model 2
    print('Training GPR model 2 ...')
    opt2 = gpf.optimizers.Scipy()
    opt2.minimize(m2.training_loss, variables=m2.trainable_variables, options=dict(maxiter=1000),
                  method="L-BFGS-B")
    # Check for nan values; if they occur reset model and use adagrad to train GPs
    if np.isnan(m2.kernel.kernels[0].lengthscales.numpy()).any():
        m2 = build_GPR_model_EPDC(X_train_n, f2_train)
        optimizer_2 = tf.keras.optimizers.Adagrad(learning_rate=0.01)
        print('Training GP model 2 using adagrad')

        @tf.function
        def step_2(i):
            optimizer_2.minimize(m2.training_loss, m2.trainable_variables)

        for i in tf.range(10000):
            step_2(i)
        NaN_flag[iter, 1] = 1
    print('Training GPR model 2 completed')

    #%% ############## Optimization of acquisition function
    # GPR models
    GPR_models = [m1, m2]

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
            loss = EEI(X_opt, GPR_models, PFn, Sigmoid_X, X_train_n, eff_length, opt=1)[0]
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
    AF_i, EEI_x_i, Prob_impr_i, YP_i, min_distance_x_i, penal_dist_x_i = EEI(x_new, GPR_models, PFn, Sigmoid_X, X_train_n, eff_length, opt=0)

    print("Iteration:", iter + 1)
    print("New design:", x_new_original[0])
    print("Initial HV:", initial_HV)
    print("Current HV:", ind_i(PF))
    print("Acquisition function:", -AF_i.numpy()[0, 0])
    print("EEI:", EEI_x_i.numpy()[0, 0])
    print("PI:", Prob_impr_i.numpy()[0, 0])
    print("Similar design penalization:", penal_dist_x_i.numpy()[0, 0])

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
num_steps_optimizer = 400  # Number of iterations of SGD-based optimizer (300 - 400)
learning_rate = 0.25  # Learning rate optimizer
n_dimensions = 3
n_samples_opt = np.min((25 * n_dimensions, 500)) # Number of points for multi-point optimization of the AF
eff_length = ((0.05 ** 2) * n_dimensions) ** 0.5  # Effective lenght for penalization of similar designs
ref_point = np.array([-6., 15.])# Reference point for calculation of dominated hypervolume
X_bounds = np.concatenate((-5*np.ones((1,n_dimensions)),5*np.ones((1,n_dimensions))),0) # Box constraints of the optimization problem

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
n_objectives = F_train.shape[1]  # number of objectives

# Arrays to track behaviour of optimization approach
HV_optimization = np.zeros((total_iter, 1))
NaN_flag = np.zeros((total_iter, 2))

# Normalization of inputs outside optimization loop as we know its maximum and minimum values
scaler_X = MinMaxScaler()
scaler_X.fit(X_bounds)

# Bayesian optimization loop
for iter in range(total_iter):
    #%% ############# Find Pareto front and Pareto set
    # Pareto designs: O if dominated, 1 if not dominated
    Pareto_index = Pareto_front(F_train)
    # Pareto front
    PF = F_train[Pareto_index]
    # Pareto designs
    PD = X_train[Pareto_index, :]
    ################## Sort Pareto front for multi-objective optimization code
    ################## that uses EEI approach (needed!!)
    PF = PF[PF[:, 0].argsort(), :]
    PD = X_train[Pareto_index, :]
    PD = PD[PF[:, 0].argsort(), :]
    ##################
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

    #%% ############## GP models
    # Training data for GPR models
    f1_train, f2_train = np.hsplit(F_train_n, n_objectives)

    # Build GPR models
    m1 = build_GPR_model_EPDC(X_train_n, f1_train)
    m2 = build_GPR_model_EPDC(X_train_n, f2_train)

    # Train GPR model 1
    print('Training GPR model 1 ...')
    opt1 = gpf.optimizers.Scipy()
    opt1.minimize(m1.training_loss, variables=m1.trainable_variables, options=dict(maxiter=1000),
                  method="L-BFGS-B")
    # Check for nan values; if they occur reset model and use adagrad to train GPs
    if np.isnan(m1.kernel.kernels[0].lengthscales.numpy()).any():
        m1 = build_GPR_model_EPDC(X_train_n, f1_train)
        optimizer_1 = tf.keras.optimizers.Adagrad(learning_rate=0.01)
        print('Training GP model 1 using adagrad')

        @tf.function
        def step_1(i):
            optimizer_1.minimize(m1.training_loss, m1.trainable_variables)
        for i in tf.range(10000):
            step_1(i)
        NaN_flag[iter, 0] = 1
    print('Training GPR model 1 completed')

    # Train GPR model 2
    print('Training GPR model 2 ...')
    opt2 = gpf.optimizers.Scipy()
    opt2.minimize(m2.training_loss, variables=m2.trainable_variables, options=dict(maxiter=1000),
                  method="L-BFGS-B")
    # Check for nan values; if they occur reset model and use adagrad to train GPs
    if np.isnan(m2.kernel.kernels[0].lengthscales.numpy()).any():
        m2 = build_GPR_model_EPDC(X_train_n, f2_train)
        optimizer_2 = tf.keras.optimizers.Adagrad(learning_rate=0.01)
        print('Training GP model 2 using adagrad')

        @tf.function
        def step_2(i):
            optimizer_2.minimize(m2.training_loss, m2.trainable_variables)

        for i in tf.range(10000):
            step_2(i)
        NaN_flag[iter, 1] = 1
    print('Training GPR model 2 completed')

    #%% ############## Optimization of acquisition function
    # GPR models
    GPR_models = [m1, m2]

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
            loss = EEI(X_opt, GPR_models, PFn, Sigmoid_X, X_train_n, eff_length, opt=1)[0]
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
    AF_i, EEI_x_i, Prob_impr_i, YP_i, min_distance_x_i, penal_dist_x_i = EEI(x_new, GPR_models, PFn, Sigmoid_X, X_train_n, eff_length, opt=0)

    print("Iteration:", iter + 1)
    print("New design:", x_new_original[0])
    print("Initial HV:", initial_HV)
    print("Current HV:", ind_i(PF))
    print("Acquisition function:", -AF_i.numpy()[0, 0])
    print("EEI:", EEI_x_i.numpy()[0, 0])
    print("PI:", Prob_impr_i.numpy()[0, 0])
    print("Similar design penalization:", penal_dist_x_i.numpy()[0, 0])

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
test_index = 0 # select a test index from 0 to 4 (initial sampling plan)
total_iter = 200 # Total number of iterations
num_steps_optimizer = 400  # Number of iterations of SGD-based optimizer (300 - 400)
learning_rate = 0.25  # Learning rate optimizer
n_dimensions = 6
n_samples_opt = np.min((25 * n_dimensions, 500)) # Number of points for multi-point optimization of the AF
eff_length = ((0.05 ** 2) * n_dimensions) ** 0.5  # Effective lenght for penalization of similar designs
ref_point = np.array([0., 0.]) # Reference point for calculation of dominated hypervolume
X_bounds = np.concatenate((np.zeros((1,n_dimensions)),6*np.ones((1,n_dimensions))),0) # Box constraints of the optimization problem

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
n_objectives = F_train.shape[1]  # number of objectives

# Arrays to track behaviour of optimization approach
HV_optimization = np.zeros((total_iter, 1))
NaN_flag = np.zeros((total_iter, 2))

# Normalization of inputs outside optimization loop as we know its maximum and minimum values
scaler_X = MinMaxScaler()
scaler_X.fit(X_bounds)

# Bayesian optimization loop
for iter in range(total_iter):
    #%% ############# Find Pareto front and Pareto set
    # Pareto designs: O if dominated, 1 if not dominated
    Pareto_index = Pareto_front(F_train)
    # Pareto front
    PF = F_train[Pareto_index]
    # Pareto designs
    PD = X_train[Pareto_index, :]
    ################## Sort Pareto front for multi-objective optimization code
    ################## that uses EEI approach (needed!!)
    PF = PF[PF[:, 0].argsort(), :]
    PD = X_train[Pareto_index, :]
    PD = PD[PF[:, 0].argsort(), :]
    ##################
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

    #%% ############## GP models
    # Training data for GPR models
    f1_train, f2_train = np.hsplit(F_train_n, n_objectives)

    # Build GPR models
    m1 = build_GPR_model_EPDC(X_train_n, f1_train)
    m2 = build_GPR_model_EPDC(X_train_n, f2_train)

    # Train GPR model 1
    print('Training GPR model 1 ...')
    opt1 = gpf.optimizers.Scipy()
    opt1.minimize(m1.training_loss, variables=m1.trainable_variables, options=dict(maxiter=1000),
                  method="L-BFGS-B")
    # Check for nan values; if they occur reset model and use adagrad to train GPs
    if np.isnan(m1.kernel.kernels[0].lengthscales.numpy()).any():
        m1 = build_GPR_model_EPDC(X_train_n, f1_train)
        optimizer_1 = tf.keras.optimizers.Adagrad(learning_rate=0.01)
        print('Training GP model 1 using adagrad')

        @tf.function
        def step_1(i):
            optimizer_1.minimize(m1.training_loss, m1.trainable_variables)
        for i in tf.range(10000):
            step_1(i)
        NaN_flag[iter, 0] = 1
    print('Training GPR model 1 completed')

    # Train GPR model 2
    print('Training GPR model 2 ...')
    opt2 = gpf.optimizers.Scipy()
    opt2.minimize(m2.training_loss, variables=m2.trainable_variables, options=dict(maxiter=1000),
                  method="L-BFGS-B")
    # Check for nan values; if they occur reset model and use adagrad to train GPs
    if np.isnan(m2.kernel.kernels[0].lengthscales.numpy()).any():
        m2 = build_GPR_model_EPDC(X_train_n, f2_train)
        optimizer_2 = tf.keras.optimizers.Adagrad(learning_rate=0.01)
        print('Training GP model 2 using adagrad')

        @tf.function
        def step_2(i):
            optimizer_2.minimize(m2.training_loss, m2.trainable_variables)

        for i in tf.range(10000):
            step_2(i)
        NaN_flag[iter, 1] = 1
    print('Training GPR model 2 completed')

    #%% ############## Optimization of acquisition function
    # GPR models
    GPR_models = [m1, m2]

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
            loss = EEI(X_opt, GPR_models, PFn, Sigmoid_X, X_train_n, eff_length, opt=1)[0]
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
    AF_i, EEI_x_i, Prob_impr_i, YP_i, min_distance_x_i, penal_dist_x_i = EEI(x_new, GPR_models, PFn, Sigmoid_X, X_train_n, eff_length, opt=0)

    print("Iteration:", iter + 1)
    print("New design:", x_new_original[0])
    print("Initial HV:", initial_HV)
    print("Current HV:", ind_i(PF))
    print("Acquisition function:", -AF_i.numpy()[0, 0])
    print("EEI:", EEI_x_i.numpy()[0, 0])
    print("PI:", Prob_impr_i.numpy()[0, 0])
    print("Similar design penalization:", penal_dist_x_i.numpy()[0, 0])

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
num_steps_optimizer = 400  # Number of iterations of SGD-based optimizer (300 - 400)
learning_rate = 0.25  # Learning rate optimizer
n_dimensions = 6
n_samples_opt = np.min((25 * n_dimensions, 500)) # Number of points for multi-point optimization of the AF
eff_length = ((0.05 ** 2) * n_dimensions) ** 0.5  # Effective lenght for penalization of similar designs
ref_point = np.array([1., 7.]) # Reference point for calculation of dominated hypervolume
X_bounds = np.concatenate((np.zeros((1,n_dimensions)),np.ones((1,n_dimensions))),0) # Box constraints of the optimization problem

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
n_objectives = F_train.shape[1]  # number of objectives

# Arrays to track behaviour of optimization approach
HV_optimization = np.zeros((total_iter, 1))
NaN_flag = np.zeros((total_iter, 2))

# Normalization of inputs outside optimization loop as we know its maximum and minimum values
scaler_X = MinMaxScaler()
scaler_X.fit(X_bounds)

# Bayesian optimization loop
for iter in range(total_iter):
    #%% ############# Find Pareto front and Pareto set
    # Pareto designs: O if dominated, 1 if not dominated
    Pareto_index = Pareto_front(F_train)
    # Pareto front
    PF = F_train[Pareto_index]
    # Pareto designs
    PD = X_train[Pareto_index, :]
    ################## Sort Pareto front for multi-objective optimization code
    ################## that uses EEI approach (needed!!)
    PF = PF[PF[:, 0].argsort(), :]
    PD = X_train[Pareto_index, :]
    PD = PD[PF[:, 0].argsort(), :]
    ##################
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

    #%% ############## GP models
    # Training data for GPR models
    f1_train, f2_train = np.hsplit(F_train_n, n_objectives)

    # Build GPR models
    m1 = build_GPR_model_EPDC(X_train_n, f1_train)
    m2 = build_GPR_model_EPDC(X_train_n, f2_train)

    # Train GPR model 1
    print('Training GPR model 1 ...')
    opt1 = gpf.optimizers.Scipy()
    opt1.minimize(m1.training_loss, variables=m1.trainable_variables, options=dict(maxiter=1000),
                  method="L-BFGS-B")
    # Check for nan values; if they occur reset model and use adagrad to train GPs
    if np.isnan(m1.kernel.kernels[0].lengthscales.numpy()).any():
        m1 = build_GPR_model_EPDC(X_train_n, f1_train)
        optimizer_1 = tf.keras.optimizers.Adagrad(learning_rate=0.01)
        print('Training GP model 1 using adagrad')

        @tf.function
        def step_1(i):
            optimizer_1.minimize(m1.training_loss, m1.trainable_variables)
        for i in tf.range(10000):
            step_1(i)
        NaN_flag[iter, 0] = 1
    print('Training GPR model 1 completed')

    # Train GPR model 2
    print('Training GPR model 2 ...')
    opt2 = gpf.optimizers.Scipy()
    opt2.minimize(m2.training_loss, variables=m2.trainable_variables, options=dict(maxiter=1000),
                  method="L-BFGS-B")
    # Check for nan values; if they occur reset model and use adagrad to train GPs
    if np.isnan(m2.kernel.kernels[0].lengthscales.numpy()).any():
        m2 = build_GPR_model_EPDC(X_train_n, f2_train)
        optimizer_2 = tf.keras.optimizers.Adagrad(learning_rate=0.01)
        print('Training GP model 2 using adagrad')

        @tf.function
        def step_2(i):
            optimizer_2.minimize(m2.training_loss, m2.trainable_variables)

        for i in tf.range(10000):
            step_2(i)
        NaN_flag[iter, 1] = 1
    print('Training GPR model 2 completed')

    #%% ############## Optimization of acquisition function
    # GPR models
    GPR_models = [m1, m2]

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
            loss = EEI(X_opt, GPR_models, PFn, Sigmoid_X, X_train_n, eff_length, opt=1)[0]
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
    AF_i, EEI_x_i, Prob_impr_i, YP_i, min_distance_x_i, penal_dist_x_i = EEI(x_new, GPR_models, PFn, Sigmoid_X, X_train_n, eff_length, opt=0)

    print("Iteration:", iter + 1)
    print("New design:", x_new_original[0])
    print("Initial HV:", initial_HV)
    print("Current HV:", ind_i(PF))
    print("Acquisition function:", -AF_i.numpy()[0, 0])
    print("EEI:", EEI_x_i.numpy()[0, 0])
    print("PI:", Prob_impr_i.numpy()[0, 0])
    print("Similar design penalization:", penal_dist_x_i.numpy()[0, 0])

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