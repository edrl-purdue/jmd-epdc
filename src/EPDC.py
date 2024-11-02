# Import packages
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

def acquisition_function_2_obj(X, GPR_models, X_train, PF, Transf_X, eff_length, n_mc_samples, gamma_dominated, gamma_pareto, opt):
        # Transformation of variables to satisfy box constraints
        if opt == 1:
            # Transformation of variable
            X = Transf_X(X)

        # Component 1: Expected distance to the closest Pareto design
        # GPR predictions
        f1_mean, f1_var = GPR_models[0].predict_y(X)
        f1_sdv = tf.sqrt(f1_var)
        f2_mean, f2_var = GPR_models[1].predict_y(X)
        f2_sdv = tf.sqrt(f2_var)

        # Monte-Carlo sampling
        tfd = tfp.distributions
        n_obj = GPR_models.__len__()

        # Mean vector and covariance matrix for MC sampling
        F_mean = tf.reshape(tf.concat([f1_mean, f2_mean], 1), [-1, n_obj])
        F_sdv = tf.reshape(tf.concat([f1_sdv, f2_sdv], 1), [-1, n_obj])

        # Instantiate MVN distribution
        MV_normal = tfd.MultivariateNormalDiag(loc=F_mean, scale_diag=F_sdv)

        # Monte Carlo samples
        F_mc = MV_normal.sample(n_mc_samples)

        # Distance to the closest Pareto design
        F_reshaped = tf.reshape(F_mc, [n_mc_samples, -1, 1, n_obj])
        distance_mc = tf.math.reduce_euclidean_norm(PF - F_reshaped, axis=3)
        min_distance = tf.math.reduce_min(distance_mc, axis=2)

        # Check Pareto optimizality condition to penalize distance of dominated designs
        # Unique designs in current Pareto front
        PF_unique = np.unique(PF, axis=0)
        # Condition 1:
        cond_1 = tf.math.reduce_all(tf.math.less_equal(PF_unique, F_reshaped), axis=3)
        # Condition 2:
        cond_2 = tf.math.reduce_any(tf.math.less(PF_unique, F_reshaped), axis=3)
        # Combine conditions (if satisfied, the design is not Pareto optimal)
        F_label = tf.math.reduce_any(tf.math.logical_and(cond_1, cond_2), axis=2)
        # If true, the design is not optimal (penalize distance)
        F_label = tf.cast(F_label, tf.float32) * (-gamma_pareto + gamma_dominated) + gamma_pareto

        # Penalized distance
        min_distance = tf.cast(min_distance, tf.float32)
        min_distance = min_distance * F_label

        # Minimum distance - expectation
        Expected_min_distance = tf.math.reduce_mean(min_distance, axis=0)
        Expected_min_distance = tf.reshape(Expected_min_distance, [-1, 1])

        # %% Component 2: Penalization of similar designs
        # Intersite distances - input space
        n_dim = X.shape[1]
        X_res = tf.reshape(X, [-1, 1, n_dim])

        # Square distance
        distance_x = tf.math.reduce_euclidean_norm(X_train - X_res, axis=2)
        min_distance_x = tf.math.reduce_min(distance_x, axis=1)
        min_distance_x = tf.cast(min_distance_x, tf.float32)

        # Transformation of the distance to prevent biasing the search towards maximization of
        # the intersite distance of the training data
        k_lenght = 2 / eff_length  # for arguments larger than 2 erf tends to 1
        penal_dist_x = tf.math.erf(min_distance_x * k_lenght)
        penal_dist_x = tf.reshape(penal_dist_x, [-1, 1])

        # Composed acquisition function
        AF = Expected_min_distance + 0.1*penal_dist_x
        return -AF, Expected_min_distance, penal_dist_x

# Three-objective functions
def acquisition_function_3_obj(X, GPR_models, X_train, PF, Transf_X, eff_length, n_mc_samples, gamma_dominated, gamma_pareto, opt):
        # Transformation of variables to satisfy box constraints
        if opt == 1:
                # Transformation of variable
                X = Transf_X(X)

        # Component 1: Expected distance to the closest Pareto design
        # GPR predictions
        f1_mean, f1_var = GPR_models[0].predict_y(X)
        f1_sdv = tf.sqrt(f1_var)
        f2_mean, f2_var = GPR_models[1].predict_y(X)
        f2_sdv = tf.sqrt(f2_var)
        f3_mean, f3_var = GPR_models[2].predict_y(X)
        f3_sdv = tf.sqrt(f3_var)

        # Monte-Carlo sampling
        tfd = tfp.distributions
        n_obj = GPR_models.__len__()

        # Mean vector and covariance matrix for MC sampling
        F_mean = tf.reshape(tf.concat([f1_mean, f2_mean, f3_mean], 1), [-1, n_obj])
        F_sdv = tf.reshape(tf.concat([f1_sdv, f2_sdv, f3_sdv], 1), [-1, n_obj])

        # Instantiate MVN distribution
        MV_normal = tfd.MultivariateNormalDiag(loc=F_mean, scale_diag=F_sdv)

        # Monte Carlo samples
        F_mc = MV_normal.sample(n_mc_samples)

        # Distance to the closest Pareto design
        F_reshaped = tf.reshape(F_mc, [n_mc_samples, -1, 1, n_obj])
        distance_mc = tf.math.reduce_euclidean_norm(PF - F_reshaped, axis=3)
        min_distance = tf.math.reduce_min(distance_mc, axis=2)

        # Check Pareto optimizality condition to penalize distance of dominated designs
        # Unique designs in current Pareto front
        PF_unique = np.unique(PF, axis=0)
        # Condition 1:
        cond_1 = tf.math.reduce_all(tf.math.less_equal(PF_unique, F_reshaped), axis=3)
        # Condition 2:
        cond_2 = tf.math.reduce_any(tf.math.less(PF_unique, F_reshaped), axis=3)
        # Combine conditions (if satisfied, the design is not Pareto optimal)
        F_label = tf.math.reduce_any(tf.math.logical_and(cond_1, cond_2), axis=2)
        # If true, the design is not optimal (penalize distance)
        F_label = tf.cast(F_label, tf.float32) * (-gamma_pareto + gamma_dominated) + gamma_pareto

        # Penalized distance
        min_distance = tf.cast(min_distance, tf.float32)
        min_distance = min_distance * F_label

        # Minimum distance - expectation
        Expected_min_distance = tf.math.reduce_mean(min_distance, axis=0)
        Expected_min_distance = tf.reshape(Expected_min_distance, [-1, 1])

        # %% Component 2: Penalization of similar designs
        # Intersite distances - input space
        n_dim = X.shape[1]
        X_res = tf.reshape(X, [-1, 1, n_dim])

        # Square distance
        distance_x = tf.math.reduce_euclidean_norm(X_train - X_res, axis=2)
        min_distance_x = tf.math.reduce_min(distance_x, axis=1)
        min_distance_x = tf.cast(min_distance_x, tf.float32)

        # Transformation of the distance to prevent biasing the search towards maximization of
        # the intersite distance of the training data
        k_lenght = 2 / eff_length  # for arguments larger than 2 erf tends to 1
        penal_dist_x = tf.math.erf(min_distance_x * k_lenght)
        penal_dist_x = tf.reshape(penal_dist_x, [-1, 1])

        # Composed acquisition function
        AF = Expected_min_distance + 0.1*penal_dist_x
        return -AF, Expected_min_distance, penal_dist_x