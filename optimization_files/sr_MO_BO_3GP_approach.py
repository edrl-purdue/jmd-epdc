import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

def Augmented_Tchebycheff(X,F,weights_tb,rho_tb,lambda_tb):
    F_scaled = F * weights_tb
    # First term
    term_1 = F_scaled.max(1).reshape(-1, 1)
    # Second term
    term_2 = rho_tb * F_scaled.sum(1).reshape(-1, 1)
    # Thrid term
    norm_2_X = np.sum(X ** 2, 1).reshape(-1, 1) ** (1. / 2)
    term_3 = lambda_tb * norm_2_X
    # Augmented_Tchebycheff
    F_aug_Tcheb = term_1 + term_2 + term_3
    return F_aug_Tcheb

def Upper_confidence_bound(X, GPR, Beta, Slack):
    f_mean, f_var = GPR.predict_y(X)
    f_std = f_var ** 0.5
    UCB = f_mean + Beta * f_std
    UCB_shift = UCB + Slack
    return UCB_shift

def Expected_improvement_classifier(X, GPC, y_best):
    tfd = tfp.distributions
    dist = tfd.Normal(loc=0., scale=1.)
    # GPC predictions
    mu_hat, varc_hat = GPC.predict_y(X)
    fc_hat = tf.cast(mu_hat, tf.float32)
    sc_hat = tf.cast(varc_hat ** 0.5, tf.float32)
    # Expected improvement
    u = (fc_hat - y_best) / sc_hat
    EI_c = (fc_hat - y_best) * dist.cdf(u) + sc_hat * dist.prob(u)
    return EI_c

def Probability_improvement_classifier(X, GPC, y_best):
    tfd = tfp.distributions
    dist = tfd.Normal(loc=0., scale=1.)
    # GPC predictions
    mu_hat, varc_hat = GPC.predict_y(X)
    fc_hat = tf.cast(mu_hat, tf.float32)
    sc_hat = tf.cast(varc_hat ** 0.5, tf.float32)
    # Probability improvement
    u = (fc_hat - y_best) / sc_hat
    PI_c = dist.prob(u)
    return PI_c

def Composed_acquisition_function(X, GPR, GPC, Beta, Slack, y_best, Transf_X, opt):
    # Transformation of variables to satisfy box constraints
    if opt == 1:
        # Transformation of variable
        X = Transf_X(X)

    # Upper confidence bound of Neg. Augm. Tchebycheff function
    UCB = Upper_confidence_bound(X, GPR, Beta, Slack)
    UCB_32 = tf.cast(UCB, tf.float32)

    # Expected improvement classifier
    EI_c = Expected_improvement_classifier(X, GPC, y_best)

    # Acquisition function
    AF = UCB_32*EI_c
    return -AF, UCB_32, EI_c


def Composed_acquisition_function_penalized(X, GPR, GPC, Beta, Slack, y_best, Transf_X, X_train, eff_length, opt):
    # Transformation of variables to satisfy box constraints
    if opt == 1:
        # Transformation of variable
        X = Transf_X(X)

    # Upper confidence bound of Neg. Augm. Tchebycheff function
    UCB = Upper_confidence_bound(X, GPR, Beta, Slack)
    UCB_32 = tf.cast(UCB, tf.float32)

    # Expected improvement classifier
    EI_c = Expected_improvement_classifier(X, GPC, y_best)

    # Penalization of similar designs
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

    # Acquisition function
    AF = UCB_32*EI_c*penal_dist_x
    return -AF, UCB_32, EI_c, penal_dist_x

def Composed_acquisition_function_penalized_no_classifier(X, GPR, Beta, Slack, Transf_X, X_train, eff_length, opt):
    # Transformation of variables to satisfy box constraints
    if opt == 1:
        # Transformation of variable
        X = Transf_X(X)

    # Upper confidence bound of Neg. Augm. Tchebycheff function
    UCB = Upper_confidence_bound(X, GPR, Beta, Slack)
    UCB_32 = tf.cast(UCB, tf.float32)

    # Penalization of similar designs
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

    # Acquisition function
    AF = UCB_32*penal_dist_x
    return -AF, UCB_32, penal_dist_x

