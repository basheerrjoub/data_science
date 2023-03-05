# code to compare the three implementations of the regression
#using the least squares method
import numpy as np

X = np.array([32,25,34,35,20,26,22,26,30,21,18,28,29,26,29,28,28,27,40,57,56,34,32,34,44,25,46,43,48,41,47,38])
Y = np.array([71,48,59,64,42,54,37,56,60,38,35,47,44,58,52,62,45,44,68,94,85,62,58,59,68,53,81,64,77,78,81,59])

def calculated_LR(x):
    return 14.58 + 1.344*x

def gradient_decent(x):
    return 14.3673 + 1.3496*x

def scikit_learn(x):
    return 14.5778 + 1.352*x

def total_error(function, X, Y):
    s = 0
    for i in range(0, 32):
        err = abs(function(X[i]) - Y[i])
        sq_err = err**2
        s += sq_err
    return s/ 32

print("LSR Manual Regression: ",total_error(calculated_LR, X, Y))
print("LSR Gradient Decent: ",total_error(gradient_decent, X, Y))
print("LSR Scikit Learn: ",total_error(scikit_learn, X, Y))