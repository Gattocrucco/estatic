import estatic2d
from matplotlib import pyplot as plt
from scipy import constants
import numpy as np

fig_single = plt.figure('cap_plate_single')
fig_single.clf()
fig_single.set_tight_layout(True)

ax_geometry, ax_charge = fig_single.subplots(2, 1, sharex=True)

epsilon = 2

# steps = 100
steps = np.concatenate([[0], np.sort(np.random.rand(98)), [1]])
s = estatic2d.ConductorSet(
    estatic2d.SegmentConductor((0,0), (10,0), steps, name='bottom', potential=-1),
    estatic2d.SegmentConductor((0,1), (10,1), steps, name='top', potential=1),
    estatic2d.RectangleDielectric(bottom_left=(0,0), sides=(10,1), segments=(10,10), epsilon_rel=epsilon)
)

lines = s.draw(ax=ax_geometry)
ax_geometry.set_title('geometry')
ax_geometry.grid(linestyle=':')
ax_geometry.set_ylabel('y [m]')

s.solve()

x = np.linspace(-1, 11, 100)
y = np.linspace(-0.5, 1.5, 50)
usage = dict(use_conductors=True, use_dielectrics=True)
rt = s.draw_potential(x, y, ax=ax_geometry, **usage)
s.draw_field(x, y, ax=ax_geometry, **usage)
ax_geometry.legend(loc='best', fontsize='small')

fig_single.colorbar(rt, ax=ax_geometry)

ax_charge.plot(s.conductors[0].centers[0], s.conductors[0].sigmas, '.', color=lines[0].get_color())
ax_charge.plot(s.conductors[1].centers[0], s.conductors[1].sigmas, '.', color=lines[1].get_color())
ax_charge.set_ylabel('charge surface density [C/m$^2$]')
ax_charge.set_xlabel('x [m]')
ax_charge.grid(linestyle=':')
ax_charge.set_title('charge density')

#############

fig = plt.figure('cap_plate')
fig.clf()
fig.set_tight_layout(True)

ax = fig.add_subplot(111)
ax.set_xlabel('ratio distance between plates/width')
ax.set_ylabel('capacitance per unit length [F/m]')

distances = np.logspace(-2, 0, 20)
width = 1

cap_theo = constants.epsilon_0 * epsilon * width / distances
cap = []
for distance in distances:
    print('computing distance {:.2g}...'.format(distance))
    # steps = 100
    steps = np.concatenate([[0], np.sort(np.random.rand(98)), [1]])
    s = estatic2d.ConductorSet(
        estatic2d.SegmentConductor((0, 0),        (width, 0),        steps, potential=1000),
        estatic2d.SegmentConductor((0, distance), (width, distance), steps, potential=1001),
        estatic2d.RectangleDielectric(bottom_left=(0,0), sides=(width, distance), segments=(10,10), epsilon_rel=epsilon)
    )
    s.solve(use_dielectrics=True)
    q1 = s.conductors[0].charge_per_unit_length
    q2 = s.conductors[1].charge_per_unit_length
    assert np.allclose(q2, -q1, atol=1e-8 * constants.epsilon_0)
    cap.append((q2 - q1) / 2)

ax.plot(distances / width, cap, '.', label='computed')
ax.plot(distances / width, cap_theo, '-', label='$\\varepsilon_0 \\cdot \\varepsilon \\cdot L/d$')
ax.set_xscale('log')
ax.set_yscale('log')
ax.legend(loc='best', fontsize='small')
ax.grid(linestyle=':')
ax.grid(linestyle=':', which='minor')

fig_single.show()
fig.show()
