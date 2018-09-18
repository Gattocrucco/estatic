import estatic2d
from matplotlib import pyplot as plt
from scipy import constants
import numpy as np

# parameters
wire_diameter = 30.48e-6 # metre
N_wire = 20

shield_width = 1.5e-3
shield_depth = 1.59e-3
N_shield = 10

###########

fig_geometry = plt.figure('coax_2d_geometry')
fig_geometry.clf()
fig_geometry.set_tight_layout(True)

ax_geometry = fig_geometry.add_subplot(111)

wire = estatic2d.Circle(center=(0,0), radius=wire_diameter / 2, segments=N_wire, name='wire', potential=1)

shield_x = np.concatenate([
    np.linspace(-shield_width / 2, shield_width / 2, 2 * N_shield),
    np.ones(N_shield - 2) * shield_width / 2,
    np.cos(np.linspace(0, np.pi, 2 * N_shield)) * shield_width / 2,
    np.ones(N_shield - 2) * -shield_width / 2
])
shield_y = np.concatenate([
    np.ones(2 * N_shield) * -shield_depth / 2,
    np.linspace(-shield_depth / 2, shield_depth / 2 - shield_width / 2, N_shield)[1:-1],
    np.sin(np.linspace(0, np.pi, 2 * N_shield)) * shield_width / 2 + shield_depth / 2 - shield_width / 2,
    np.linspace(shield_depth / 2 - shield_width / 2, -shield_depth / 2, N_shield)[1:-1],
])

shield = estatic2d.Conductor(shield_x, shield_y, name='shield')

s = estatic2d.ConductorSet(wire, shield)

s.draw(ax=ax_geometry)
ax_geometry.legend(loc='best', fontsize='small')
ax_geometry.set_xlabel('x [m]')
ax_geometry.set_ylabel('y [m]')
ax_geometry.grid(linestyle=':')

s.solve()

cap = wire.charge_per_unit_length # (potential is 1)
cap_appr = 2 * np.pi * constants.epsilon_0 / np.log(shield_width / wire_diameter)

print('capacitance [pF/cm]: computed %.3g, appr. cylinder %.3g' % (cap * 1e10, cap_appr * 1e10))

###########

fig_align = plt.figure('coax_2d_alignment')
fig_align.clf()
fig_align.set_tight_layout(True)

ax_align_x, ax_align_y = fig_align.subplots(1, 2, sharey=True)

centers_x = np.linspace(-0.4e-3, 0, 10)
centers_y = np.linspace(-0.4e-3, 0.3e-3, 18)

centers_x = np.outer(centers_x, np.ones(len(centers_y)))
centers_y = np.outer(np.ones(len(centers_x)), centers_y)

capacitances = np.empty(centers_x.shape)

for i in np.ndindex(*capacitances.shape):
    this_wire = estatic2d.Circle(center=(centers_x[i], centers_y[i]), radius=wire_diameter / 2, segments=N_wire, potential=1)
    this_s = estatic2d.ConductorSet(this_wire, shield)
    this_s.solve()
    capacitances[i] = this_wire.charge_per_unit_length

color_min = 0
color_max = 0.8
for i in range(centers_x.shape[1]):
    color = color_min + (color_max - color_min) * i / (centers_x.shape[1] - 1)
    label = 'y = %.3g' % centers_y[0,i]
    ax_align_x.plot(centers_x[:,i], capacitances[:,i], '.-', color=[color] * 3, label=label, zorder=100-i)
ax_align_x.legend(loc='best', fontsize='small')
ax_align_x.set_xlabel('x [m]')
ax_align_x.set_ylabel('cap. per unit length [F/m]')
ax_align_x.grid(linestyle=':')

for i in range(centers_x.shape[0]):
    color = color_min + (color_max - color_min) * i / (centers_x.shape[0] - 1)
    label = 'x = %.3g' % centers_x[i,0]
    ax_align_y.plot(centers_y[i,:], capacitances[i,:], '.-', color=[color] * 3, label=label, zorder=100-i)
ax_align_y.legend(loc='best', fontsize='small')
ax_align_y.set_xlabel('y [m]')
ax_align_y.grid(linestyle=':')

fig_geometry.show()
fig_align.show()
