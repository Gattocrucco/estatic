import sympy
from scipy import special
from matplotlib import pyplot as plt
import numpy as np

# integral (dx+e)/(ax^2+bx+c)

fig = plt.figure('integral_1')
fig.clf()

ax = fig.add_subplot(111)

x = np.linspace(-10, 10, 500)

a = 1
b = 2
c = 1
d = 1
e = 1 # (1-np.sqrt(3))/2
f = (-b + np.sqrt(4*a*c-b**2)) / (2*a)

dy = (d*x + e) / (a * x**2 + b*x + c)

y = d/(2*a) * np.log(a * x**2 + b*x + c) + (2*e - b*d/a) / np.sqrt(4*a*c-b**2) * np.arctan((2*a*x + b) / np.sqrt(4*a*c-b**2))

y_alt = d/a * np.log(np.abs(x - f)) + (d*f + e) / (a * (x - f))

num_dy = np.diff(y) / np.diff(x)
num_dy_alt = np.diff(y_alt) / np.diff(x)

ax.plot(x, dy, label='dy', linewidth=5)
ax.plot((x[1:] + x[:-1]) / 2, num_dy, label='num. dy', linewidth=3)
ax.plot((x[1:] + x[:-1]) / 2, num_dy_alt, label='num. dy alt.', linewidth=1)
ax.legend()
ax.grid()
all_y = np.concatenate([dy, num_dy, num_dy_alt])
all_y = all_y[(all_y != 0) & np.isfinite(all_y)]
ax.set_yscale('symlog', linthreshy=np.min(np.abs(all_y)))

fig.show()
