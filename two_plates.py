from matplotlib.pyplot import *
import estatic2d
import numpy as np

figure('two_plates')
clf()

plate1 = estatic2d.SegmentConductor(
    endpoint_A=(-2.5, 0),
    endpoint_B=(-1.5, 0),
    name='left plate',
    potential=0
)

plate2 = estatic2d.SegmentConductor(
    endpoint_A=(1.5, 1),
    endpoint_B=(3.5, 3),
    name='right plate',
    potential=1
)

s = estatic2d.ConductorSet(plate1, plate2)

s.solve()

grid()

s.draw()

x = np.linspace(-1.4, 1.4, 21)
y = np.linspace(-1, 2, 31)
s.draw_potential(x, y)
x, y = np.meshgrid(x, y)
s.draw_field(x, y, scale='linear')

legend()
show()

