import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


x = np.array([32,25,34,35,20,26,22,26,30,21,18,28,29,26,29,28,28,27,40,57,56,34,32,34,44,25,46,43,48,41,47,38]).reshape(-1, 1)
y = np.array([71,48,59,64,42,54,37,56,60,38,35,47,44,58,52,62,45,44,68,94,85,62,58,59,68,53,81,64,77,78,81,59]).reshape(-1, 1)



X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.1)


reg = LinearRegression()
reg.fit(X_train, y_train)
print("W0: ", reg.intercept_)
print("W1: ", reg.coef_)

def predict(w1, w0, mid):
    return w0 + w1*mid
vals = [10, 15, 20, 30, 14]
yvals = []
for val in vals:
    print(f"Predicted for {val}:", predict(reg.coef_, reg.intercept_, val));
    yvals.append(predict(reg.coef_, reg.intercept_, val))

