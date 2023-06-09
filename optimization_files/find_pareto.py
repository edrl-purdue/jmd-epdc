# Import Python libraries
import numpy as np

def Pareto_front(B):
    """
    :param B: An (n_points, n_obj) array
    :return: A (n_points, ) boolean array, indicating whether each point is a Pareto design
    """
    Pareto_index = np.ones(B.shape[0], dtype = bool)
    for i, design_i in enumerate(B):
        # Non-repeated unique designs to determine Pareto optimality
        idx = abs((B[:, np.newaxis, :] - design_i)).sum(axis=2) == 0
        B_unique = B[~idx.flatten()]
        # Condition 1
        cond_1 = np.all(B_unique <= design_i, axis=1)
        # Condition 2
        cond_2 = np.any(B_unique < design_i, axis=1)
        # Combine conditions (if satisfied, the design is not Pareto optimal)
        Pareto_index[i] = ~np.any(cond_1 & cond_2)
    return Pareto_index