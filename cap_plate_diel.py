import estatic2d
from matplotlib import pyplot as plt
from scipy import constants
import numpy as np

fig = plt.figure('cap_plate_diel')
fig.clf()
fig.set_tight_layout(True)

ax = fig.add_subplot(111)
s = estatic2d.ConductorSet(
    estatic2d.Segment(endpoint_A=(0,0), endpoint_B=(1,0), segments=100, name='bottom plate', potential=0),
    estatic2d.Segment(endpoint_A=(0,0.1), endpoint_B=(1,0.1), segments=100, name='top plate', potential=1),
    estatic2d.Rectangle(bottom_left=(0,0.01), sides=(1,0.08), segments=(10,10), epsilon_rel=2)
)

s.draw(ax=ax)

ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.legend(loc='upper right', fontsize='small')

fig.show()
