import numpy as np
import matplotlib.pyplot as plt

def gd(x, y, iterations, learning_rate = 0.0001):
     
    w1 = 0.1
    w0 = 0.01
    costs = []
    weights = []
    prev_error = None

    for i in range(iterations):
        
        predict = (w1 * x) + w0
        err  = np.sum((y-predict)**2) / 32
  
        if prev_error and abs(prev_error-err)<=0.0000001:
            break
         
        prev_error = err
        costs.append(err)
        weights.append(w1)
         
        w1_der = -(2/32) * sum(x * (y-predict))
        w0_der = -(2/32) * sum(y-predict)
         
        w1 = w1 - (learning_rate * w1_der)
        w0 = w0 - (learning_rate * w0_der)
    
    return w1, w0

X = np.array([32,25,34,35,20,26,22,26,30,21,18,28,29,26,29,28,28,27,40,57,56,34,32,34,44,25,46,43,48,41,47,38])
Y = np.array([71,48,59,64,42,54,37,56,60,38,35,47,44,58,52,62,45,44,68,94,85,62,58,59,68,53,81,64,77,78,81,59])

w1, w0 = gd(X, Y, iterations=10000000)
print(f"W1: {w1}\nW0: {w0}")

F = w1*X + w0

plt.scatter(X, Y, marker='o', color='red')
plt.plot([min(X), max(X)], [min(F), max(F)], color='blue',markerfacecolor='green')
plt.xlabel("Mid")
plt.ylabel("Final")
plt.show()
 