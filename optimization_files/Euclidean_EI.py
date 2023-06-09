# Code to calculate the Euclidean-based expected improvement
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
# Import Gaussian distribution
tfd = tfp.distributions
norm_dist = tfd.Normal(loc=0., scale=1.)

# Probability of improvement
def probability_improvement(X,GPR_m1,GPR_m2,PF):
    # GP predictions
    f1_hat, var1_hat = GPR_m1.predict_y(X)
    s1_hat = var1_hat**0.5
    f2_hat, var2_hat = GPR_m2.predict_y(X)
    s2_hat = var2_hat**0.5
    # Convert to float 32 to enable use of norm_dist
    f1_hat = tf.cast(f1_hat, tf.float32)
    f2_hat = tf.cast(f2_hat, tf.float32)
    s1_hat = tf.cast(s1_hat, tf.float32)
    s2_hat = tf.cast(s2_hat, tf.float32)

    # Probability of improvement (Following Eq. 9.4 Forrester's book)
    # The Pareto designs are indexed (sorted) from left to right: 1, 2 , ..., m
    m = PF.shape[0]
    y1_star = PF[0:1,:]; ym_star = PF[m-1:m,:]
    y11 = y1_star[:,0:1]; y12 = y1_star[:,1:2];
    ym1 = ym_star[:,0:1]; ym2 = ym_star[:,1:2];

    term_1 = norm_dist.cdf((y11 - f1_hat) / s1_hat) # Standar normal cdf

    term_2 = 0
    for i in range(0,m-1):
        y1i_plus = PF[i+1:i+2,0:1]; y1i = PF[i:i+1,0:1]; y2i_plus = PF[i+1:i+2,1:2]
        term_2 = term_2 + ((norm_dist.cdf((y1i_plus - f1_hat) / s1_hat) - norm_dist.cdf((y1i - f1_hat) / s1_hat)) * norm_dist.cdf((y2i_plus - f2_hat) / s2_hat))

    term_3 = (1 - norm_dist.cdf((ym1 - f1_hat) / s1_hat)) * norm_dist.cdf((ym2 - f2_hat) / s2_hat)

    PI = term_1 + term_2 + term_3 + 10**-10 # Add small perturbation to prevent instabilities
    return PI, f1_hat, f2_hat, s1_hat, s2_hat

#  Centroid probability of improvement (corrected version of Forrester's book)
def centroid_probability_improvement(X,GPR_m1,GPR_m2,PF):
    m = PF.shape[0];
    y1_star = PF[0:1,:]; ym_star = PF[m-1:m,:]
    y11 = y1_star[:,0:1]; y12 = y1_star[:,1:2];
    ym1 = ym_star[:,0:1]; ym2 = ym_star[:,1:2];
    # Probability of improvement and GP predictions
    PI, f1_hat, f2_hat, s1_hat, s2_hat = probability_improvement(X, GPR_m1, GPR_m2, PF)

    # Centroid - y1 axis
    term_y11 = f1_hat * norm_dist.cdf((y11 - f1_hat) / s1_hat) - s1_hat * norm_dist.prob((y11 - f1_hat) / s1_hat)

    term_y12 = 0
    for i in range(0,m-1):
        y1i_plus = PF[i+1:i+2,0:1]; y1i = PF[i:i+1,0:1]; y2i_plus = PF[i+1:i+2,1:2]
        term_y12 = term_y12 + (f1_hat * (norm_dist.cdf((y1i_plus - f1_hat) / s1_hat) - norm_dist.cdf((y1i - f1_hat) / s1_hat)) + \
                               s1_hat * (norm_dist.prob((y1i - f1_hat) / s1_hat) - norm_dist.prob((y1i_plus - f1_hat) / s1_hat))) * norm_dist.cdf((y2i_plus - f2_hat) / s2_hat)

    term_y13 = (f1_hat * (1 - norm_dist.cdf((ym1 - f1_hat) / s1_hat)) + \
                s1_hat * (norm_dist.prob((ym1 - f1_hat) / s1_hat))) * norm_dist.cdf((ym2 - f2_hat) / s2_hat) # Eq. 9.7  has a different term

    YPI_1 = (term_y11 + term_y12 + term_y13)/PI

    # Centroid - y2 axis (switch PF order to use same equations for y1 axis)
    # Order PF using objective 1 (order from bottom to top)
    PF = PF[PF[:, 1].argsort(), :]
    y1_star = PF[0:1, :];
    ym_star = PF[m - 1:m, :]
    y11 = y1_star[:, 0:1];
    y12 = y1_star[:, 1:2];
    ym1 = ym_star[:, 0:1];
    ym2 = ym_star[:, 1:2];

    # Centroid - y2 axis
    term_y21 = f2_hat * norm_dist.cdf((y12 - f2_hat) / s2_hat) - s2_hat * norm_dist.prob((y12 - f2_hat) / s2_hat)

    term_y22 = 0
    for i in range(0, m - 1):
        y2i_plus = PF[i + 1:i + 2, 1:2]; y2i = PF[i:i + 1, 1:2]; y1i_plus = PF[i + 1:i + 2, 0:1]
        term_y22 = term_y22 + (f2_hat * (norm_dist.cdf((y2i_plus - f2_hat) / s2_hat) - norm_dist.cdf((y2i - f2_hat) / s2_hat)) + \
                               s2_hat * (norm_dist.prob((y2i - f2_hat) / s2_hat) - norm_dist.prob((y2i_plus - f2_hat) / s2_hat))) * norm_dist.cdf((y1i_plus - f1_hat) / s1_hat)

    term_y23 = (f2_hat * (1 - norm_dist.cdf((ym2 - f2_hat) / s2_hat)) + \
                s2_hat * (norm_dist.prob((ym2 - f2_hat) / s2_hat))) * norm_dist.cdf(
        (ym1 - f1_hat) / s1_hat)  # Eq. 9.7  has a different term

    YPI_2 = (term_y21 + term_y22 + term_y23) / PI

    YP = tf.concat([YPI_1, YPI_2], axis=1)
    return YP

def Euclidean_MEI(YP,PF,PI):
    n_obj = YP.shape[1]
    YP_res = tf.reshape(YP, [-1, 1, n_obj])
    distance = tf.math.reduce_euclidean_norm(PF - YP_res, axis=2)
    min_distance = tf.math.reduce_min(distance, axis=1)
    min_distance = tf.cast(min_distance, tf.float32)
    min_distance = tf.reshape(min_distance, [-1, 1])
    EEI = min_distance * PI
    return EEI, min_distance


def EEI(X,GPR_models,PF,Transf_X,opt):
    # Transformation of variables to satisfy box constraints during gradient based optimization
    # supported by AD
    if opt == 1:
        # Transformation of design variables
        X = Transf_X(X)

    # GPR models
    m1 = GPR_models[0]
    m2 = GPR_models[1]

    # Probability of improvement
    Prob_impr = probability_improvement(X, m1, m2, PF)[0]
    # Centroid coordinates
    YP = centroid_probability_improvement(X, m1, m2, PF)
    # Euclidean-based Expected improvement
    EEI_x, min_distance_x = Euclidean_MEI(YP, PF, Prob_impr)

    if opt == 1:
        # Minimization of the -EI
        return -EEI_x, Prob_impr, YP, min_distance_x
    else:
        # Visualization
        return EEI_x, Prob_impr, YP, min_distance_x

# %% Penalized expected improvement using Tensorflow native functions to enable Automatic Differentiation (AD)
def Penalized_EEI(X,GPR_models,PF,Transf_X, X_train_n, eff_length,opt):
    # Transformation of variables to satisfy box constraints during gradient based optimization
    # supported by AD
    if opt == 1:
        # Transformation of design variables
        X = Transf_X(X)

    # GPR models
    m1 = GPR_models[0]
    m2 = GPR_models[1]

    # Probability of improvement
    Prob_impr = probability_improvement(X, m1, m2, PF)[0]
    # Centroid coordinates
    YP = centroid_probability_improvement(X, m1, m2, PF)
    # Euclidean-based Expected improvement
    EEI_x, min_distance_x = Euclidean_MEI(YP, PF, Prob_impr)

    # Minimum intersite distances to prevent re-sampling
    n_dim = X.shape[1]
    X_res = tf.reshape(X, [-1, 1, n_dim])
    distance = tf.math.reduce_euclidean_norm(X_train_n - X_res, axis=2)
    min_distance = tf.math.reduce_min(distance, axis=1)
    min_distance = tf.cast(min_distance, tf.float32)
    k_lenght = 2 / eff_length  # for arguments larger than 2 erf tends to 1
    penal_dist_x = tf.math.erf(min_distance* k_lenght)
    penal_dist_x = tf.reshape(penal_dist_x, [-1, 1])

    # Composed acquisition function
    AF = EEI_x * penal_dist_x

    return -AF, EEI_x, Prob_impr, YP, min_distance_x, penal_dist_x