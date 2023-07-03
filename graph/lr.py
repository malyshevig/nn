import time

import numpy as np
import random
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

X = np.array([[0],[1],[2],[3]])

random.seed(time.time_ns())

#X = np.array([[410, 1468680199], [150, 525443902], [, 2], [2, 3]])
# y = 1 * x_0 + 2 * x_1 + 3
z = np.array([[3+2*random.random()], [3+2*random.random()], [3+2*random.random()], [3+2*random.random()]])

y = np.dot(X, 2) +z

X = np.array([[410],      [150],      [205],      [370],       [195]])
y= np.array([[1468680199],[525443902],[654116072],[1095780321],[613688695]])

reg = LinearRegression().fit(X, y)

reg.score(X, y)
print (f"{reg.coef_}, { reg.intercept_}")


#reg.predict(np.array([[10]]))

plt.scatter(X, y, color="red")

x1 = [[x] for x in np.linspace(0, 500, 10)]
y1 = reg.predict(x1)


plt.plot(x1, y1, color="blue", linewidth=2)

#plt.plot(X, y, color="blue", linewidth=2)
plt.show()