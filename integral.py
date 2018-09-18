import sympy
from scipy import special
from matplotlib import pyplot as plt
import numpy as np

fig = plt.figure('integral')
fig.clf()

ax = fig.add_subplot(111)

x = np.linspace(0.1, 10, 500)

a = 2.2345

dy = 1/x * np.arctan(a/x)

y = 1j/2 * (special.spence(1 - 1j*a/x) - special.spence(1 + 1j*a/x))
assert np.allclose(np.imag(y), 0)

ax.plot(x, dy, label='dy')
ax.plot((x[1:] + x[:-1]) / 2, np.diff(y) / np.diff(x), label='num. dy')
ax.legend()
ax.grid()
ax.set_yscale('log')

fig.show()

u1 = 1
u2 = 2
v1 = 3
v2 = 4

f = lambda u, v: 1j/2 * (special.spence(1 - 1j*v/u) - special.spence(1 + 1j*v/u))
g = lambda v, u: f(u, v)

ia = (f(u2, v2) - f(u2, v1)) - (f(u1, v2) - f(u1, v1))
ib = (g(u2, v2) - g(u2, v1)) - (g(u1, v2) - g(u1, v1))
