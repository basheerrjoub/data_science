import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


mid = np.array([32,25,34,35,20,26,22,26,30,21,18,28,29,26,29,28,28,27,40,57,56,34,32,34,44,25,46,43,48,41,47,38])
final = np.array([71,48,59,64,42,54,37,56,60,38,35,47,44,58,52,62,45,44,68,94,85,62,58,59,68,53,81,64,77,78,81,59])


#fit data to the model
model = LinearRegression(normalize=True)
x = np.expand_dims(mid, 1)
y = final

model.fit(x,y)

print("w1: ", model.coef_, "w0: ", model.intercept_)

plt.scatter(mid, final)
x = mid
f = model.coef_*x + model.intercept_
plt.plot(x, f, 'r')
plt.xlabel("Midterm marks")
plt.ylabel("Final marks")
plt.show()