# Import external packages
import numpy as np

########## Two-objective test problems
# Define test function
def ZDT1(x, dim):
    n = np.shape(x)[0]  # Number of samples
    # Function 1
    f1 = x[:,0:1]
    # Function 2
    term_1 = np.zeros((n, 1))
    for i in range(1, dim):
        term_1 = term_1 + x[:,i:i+1]/(dim-1)
    g = 1 + 9*term_1
    h = 1 - np.sqrt(f1/g)
    f2 = g*h
    # Test functions
    f = np.concatenate((f1, f2), axis=1)
    return f

def ZDT2(x, dim):
    n = np.shape(x)[0]  # Number of samples
    # Function 1
    f1 = x[:,0:1]
    # Function 2
    term_1 = np.zeros((n, 1))
    for i in range(1, dim):
        term_1 = term_1 + x[:,i:i+1]/(dim-1)
    g = 1 + 9*term_1
    h = 1 - (f1/g)**2
    f2 = g*h
    # Test functions
    f = np.concatenate((f1, f2), axis=1)
    return f


def ZDT3(x, dim):
    n = np.shape(x)[0]  # Number of samples
    # Function 1
    f1 = x[:,0:1]
    # Function 2
    term_1 = np.zeros((n, 1))
    for i in range(1, dim):
        term_1 = term_1 + x[:,i:i+1]/(dim-1)
    g = 1 + 9*term_1
    h = 1 - np.sqrt(f1/g) - ((f1/g)*np.sin(10*np.pi*f1))
    f2 = g*h
    # Test functions
    f = np.concatenate((f1, f2), axis=1)
    return f

# Define test function
def Viennet(x):
    x1 = x[:,0:1]; x2 = x[:,1:2]
    # Functions
    f1 = 0.5*(x1**2 + x2**2) + np.sin(x1**2 + x2**2)
    f2 = ((3*x1 - 2*x2 +4)**2)/8 + ((x1-x2+1)**2)/27 +15
    f3 = 1/(x1**2 + x2**2 +1) - 1.1*np.exp(-(x1**2 + x2**2))
    f = np.concatenate((f1, f2, f3), axis=1)
    return f

def Kursawe(x):
    d = np.shape(x)[1] # Number of dimensions
    n = np.shape(x)[0] # Number of samples
    # Function 1
    sum_1 = np.zeros((n,1))
    for i in range(d-1):
        sum_term_1 = -10*np.exp(-0.2*np.sqrt( x[:,i:i+1]**2 + x[:,i+1:i+2]**2 ))
        sum_1 = sum_1 + sum_term_1
    f1 = sum_1
    # Function 2
    sum_2 = np.zeros((n,1))
    for i in range(d):
        sum_term_2 = np.abs(x[:,i:i+1])**0.8 + 5*np.sin(x[:,i:i+1]**3)
        sum_2 = sum_2 + sum_term_2
    f2 = sum_2
    f = np.concatenate((f1, f2), axis=1)
    return f

def Fonseca(x, dim):
    d = dim # Number of dimensions
    n = np.shape(x)[0] # Number of samples
    # Function 1
    sum_1 = np.zeros((n,1))
    for i in range(d):
        sum_term_1 = (x[:,i:i+1] - 1/np.sqrt(d))**2
        sum_1 = sum_1 + sum_term_1
    f1 = 1 - np.exp(-sum_1)
    # Function 2
    sum_2 = np.zeros((n,1))
    for i in range(d):
        sum_term_2 = (x[:,i:i+1] + 1/np.sqrt(d))**2
        sum_2 = sum_2 + sum_term_2
    f2 = 1 - np.exp(-sum_2)
    f = np.concatenate((f1, f2), axis=1)
    return f
