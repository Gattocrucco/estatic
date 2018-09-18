import estatic2d
from matplotlib import pyplot as plt
from scipy import constants
import numpy as np

fig_single = plt.figure('cap2d_cylinder_single')
fig_single.clf()
fig_single.set_tight_layout(True)

ax_geometry, ax_charge = fig_single.subplots(2, 1)

angles = 2 * np.pi * np.random.rand(200)
s = estatic2d.ConductorSet(
    estatic2d.CircleConductor((0,0), 1, np.sort(angles[:100]), name='inner', potential=1),
    estatic2d.CircleConductor((0,0), 2, np.sort(angles[100:]), name='outer', potential=0),
    estatic2d.RectangleDielectric((-2,-2), (4,4), (20,20), epsilon_rel=2)
)

lines = s.draw(ax=ax_geometry)
ax_geometry.set_title('geometry')
ax_geometry.grid(linestyle=':')
ax_geometry.legend(loc='best', fontsize='small')
ax_geometry.set_ylabel('y [m]')
ax_geometry.set_xlabel('x [m]')

s.solve()

ax_charge.plot(np.degrees(np.arctan2(*s.conductors[0].centers)), s.conductors[0].sigmas, '.', color=lines[0].get_color())
ax_charge.plot(np.degrees(np.arctan2(*s.conductors[1].centers)), s.conductors[1].sigmas, '.', color=lines[1].get_color())
ax_charge.set_ylabel('charge surface density [C/m$^2$]')
ax_charge.set_xlabel('angle [deg]')
ax_charge.grid(linestyle=':')
ax_charge.set_title('charge density')

#################

fig = plt.figure('cap2d_cylinder')
fig.clf()
fig.set_tight_layout(True)

ax_radii, ax_offset = fig.subplots(2, 1)

ax_radii.set_title('changing radius')
ax_radii.set_xlabel('ratio of radii outer/inner [m]')
ax_radii.set_ylabel('capacitance per unit length [F/m]')

outer_radii = np.logspace(0.1, 2, 20)
inner_radius = 1
epsilon = 2

cap_theo = 2 * np.pi * constants.epsilon_0 * epsilon / np.log(outer_radii / inner_radius)
cap = []
for outer_radius in outer_radii:
    print('computing outer radius {:.2g}...'.format(outer_radius))
    angles = 2 * np.pi * np.random.rand(200)
    s = estatic2d.ConductorSet(
        estatic2d.CircleConductor((0, 0), outer_radius, np.sort(angles[:100]), potential=2),
        estatic2d.CircleConductor((0, 0), inner_radius, np.sort(angles[100:]), potential=3),
        estatic2d.RectangleDielectric((-outer_radius,-outer_radius), (2*outer_radius,2*outer_radius), (20,20), epsilon_rel=epsilon)
    )
    s.solve()
    q1 = s.conductors[0].charge_per_unit_length
    q2 = s.conductors[1].charge_per_unit_length
    assert np.allclose(q2, -q1, atol=1e-8 * constants.epsilon_0)
    cap.append((q2 - q1) / 2)

ax_radii.plot(outer_radii / inner_radius, cap, '.', label='computed')
ax_radii.plot(outer_radii / inner_radius, cap_theo, '-', label='$2\\pi\\varepsilon_0\\varepsilon / \\log(R/r)$')
ax_radii.set_xscale('log')
ax_radii.set_yscale('log')
ax_radii.legend(loc='best', fontsize='small')
ax_radii.grid(linestyle='--', which='major')
ax_radii.grid(linestyle=':', which='minor')

#################

ax_offset.set_title('changing alignment')
ax_offset.set_xlabel('ratio offset/radii difference')
ax_offset.set_ylabel('capacitance per unit length [F/m]')

outer_radius = 1
inner_radius = 0.4
offsets = np.linspace(0, (outer_radius - inner_radius) * 0.9, 20)
epsilon = 2

cap_theo = 2 * np.pi * constants.epsilon_0 * epsilon / np.arccosh((outer_radius**2 + inner_radius**2 - offsets**2) / (2 * outer_radius * inner_radius))
cap_appr = 2 * np.pi * constants.epsilon_0 * epsilon / (np.log(outer_radius / inner_radius) - offsets**2 / (outer_radius**2 - inner_radius**2))
cap = []
for offset in offsets:
    print('computing offset {:.2g}...'.format(offset))
    angles = 2 * np.pi * np.random.rand(200)
    s = estatic2d.ConductorSet(
        estatic2d.CircleConductor((offset, 0), outer_radius, np.sort(angles[:100]), potential=2),
        estatic2d.CircleConductor((0, 0), inner_radius, np.sort(angles[100:]), potential=3),
        estatic2d.RectangleDielectric((-outer_radius,-outer_radius), (2*outer_radius,2*outer_radius), (20,20), epsilon_rel=epsilon)
    )
    s.solve()
    q1 = s.conductors[0].charge_per_unit_length
    q2 = s.conductors[1].charge_per_unit_length
    assert np.allclose(q2, -q1, atol=1e-8 * constants.epsilon_0)
    cap.append((q2 - q1) / 2)

ax_offset.plot(offsets / (outer_radius - inner_radius), cap, '.', label='computed')
ax_offset.plot(offsets / (outer_radius - inner_radius), cap_theo, '-', label='$2\\pi\\varepsilon_0\\varepsilon / \\mathrm{acosh}((R^2+r^2-\\Delta^2)/(2Rr))$')
ax_offset.plot(offsets / (outer_radius - inner_radius), cap_appr, '-', label='$2\\pi\\varepsilon_0\\varepsilon / (\\log(R/r) - \\Delta^2/(R^2-r^2))$')
ax_offset.legend(loc='best', fontsize='small')
ax_offset.grid(linestyle='--', which='major')
ax_offset.grid(linestyle=':', which='minor')

fig_single.show()
fig.show()
