import estatic2d
from matplotlib import pyplot as plt
import numpy as np

###### parameters ######
# all units MKS
housing_detector_top_margin = 1e-3
housing_detector_bottom_margin = 5.0e-3
housing_detector_left_margin = 5.0e-3
housing_detector_right_margin = 5.0e-3
housing_potential = 0.0

detector_height = 30.0e-3
detector_width = 5.0e-3
detector_epsilon = 11.7

electrode_spacing = 500e-6 # center-center
electrode_hi_width = 100e-6
electrode_lo_width = 40e-6
electrode_hi_potential = 50.0
electrode_lo_potential = 47.0
########################

fig = plt.figure('ihv')
fig.clf()
fig.set_tight_layout(True)

ax = fig.add_subplot(111)

###### defines objects ######

print('defining objects...')

detector_bottom_left = np.array([0, 0])

detector_bulk = estatic2d.RectangleDielectric(
    bottom_left=(detector_bottom_left[0], detector_bottom_left[1] + electrode_spacing),
    sides=(detector_width, detector_height - 2 * electrode_spacing),
    segments=(10, 10),
    name='detector',
    epsilon_rel=detector_epsilon
)

detector_bottom = estatic2d.RectangleDielectric(
    bottom_left=detector_bottom_left,
    sides=(detector_width, electrode_spacing),
    segments=(50, 5)
)

detector_top = estatic2d.RectangleDielectric(
    bottom_left=(detector_bottom_left[0], detector_bottom_left[1] + detector_height - electrode_spacing),
    sides=(detector_width, 0.9 * electrode_spacing),
    segments=(100, 9)
)

detector_toptop = estatic2d.RectangleDielectric(
    bottom_left=(detector_bottom_left[0], detector_bottom_left[1] + detector_height - 0.1 * electrode_spacing),
    sides=(detector_width, 0.1 * electrode_spacing),
    segments=(300, 3)
)

# keep detector_bulk first!
detector = detector_bulk + detector_top + detector_toptop + detector_bottom

housing = estatic2d.RectangleConductor(
    bottom_left=detector_bottom_left - np.array([housing_detector_left_margin, housing_detector_bottom_margin]),
    sides=(
        detector_width + housing_detector_left_margin + housing_detector_right_margin,
        detector_height + housing_detector_top_margin + housing_detector_bottom_margin
    ),
    segments=(100, 10),
    name='housing',
    potential=housing_potential
)

hi_electrodes = []
draw_kw = dict(
    color='red',
)
x = detector_bottom_left[0] + electrode_spacing
labeled = True
while x + electrode_spacing * 0.9 <= detector_bottom_left[0] + detector_width:
    hi_electrodes.append(estatic2d.SegmentConductor(
        center=(x, detector_bottom_left[1] + detector_height),
        vector_A_to_B=(electrode_hi_width, 0),
        segments=10,
        name='electrode',
        potential=electrode_hi_potential,
        draw_kwargs=draw_kw
    ))
    hi_electrodes.append(estatic2d.SegmentConductor(
        center=(x, detector_bottom_left[1]),
        vector_A_to_B=(electrode_hi_width, 0),
        segments=10,
        name='electrode',
        potential=-electrode_hi_potential,
        draw_kwargs=draw_kw
    ))
    x += 2 * electrode_spacing
    if labeled:
        draw_kw = dict(**draw_kw)
        draw_kw.update(label=None)
        labeled = False

lo_electrodes = []
draw_kw = dict(
    color='blue',
)
x = detector_bottom_left[0] + 2 * electrode_spacing
labeled = True
while x + electrode_spacing * 0.9 <= detector_bottom_left[0] + detector_width:
    lo_electrodes.append(estatic2d.SegmentConductor(
        center=(x, detector_bottom_left[1] + detector_height),
        vector_A_to_B=(electrode_lo_width, 0),
        segments=10,
        name='electrode',
        potential=electrode_lo_potential,
        draw_kwargs=draw_kw
    ))
    lo_electrodes.append(estatic2d.SegmentConductor(
        center=(x, detector_bottom_left[1]),
        vector_A_to_B=(electrode_lo_width, 0),
        segments=10,
        name='electrode',
        potential=-electrode_lo_potential,
        draw_kwargs=draw_kw
    ))
    x += 2 * electrode_spacing
    if labeled:
        draw_kw = dict(**draw_kw)
        draw_kw.update(label=None)
        labeled = False

s = estatic2d.ConductorSet(detector, housing, *hi_electrodes, *lo_electrodes)

###### compute ######

print('computing...')
s.solve()

###### draw ######

print('drawing...')
# margin = 1e-3
# x = np.linspace(
#     detector_bottom_left[0] - housing_detector_left_margin - margin,
#     detector_bottom_left[0] + detector_width + housing_detector_right_margin + margin,
#     50
# )
# y = np.linspace(
#     detector_bottom_left[1] - housing_detector_bottom_margin - margin,
#     detector_bottom_left[1] + detector_height + housing_detector_top_margin + margin,
#     50
# )
# x = np.linspace(
#     detector_bottom_left[0] - 0.02 * detector_width,
#     detector_bottom_left[0] + 0.12 * detector_width,
#     50
# )
# y = np.linspace(
#     detector_bottom_left[1] + 0.68 * detector_height,
#     detector_bottom_left[1] + 0.82 * detector_height,
#     50
# )
# x = np.linspace(
#     detector_bottom_left[0] + 1.9 * electrode_spacing,
#     detector_bottom_left[0] + 3.1 * electrode_spacing,
#     50
# )
# y = np.linspace(
#     detector_bottom_left[1] + detector_height - electrode_spacing / 4,
#     detector_bottom_left[1] + detector_height + electrode_spacing / 4,
#     50
# )
# s.draw_potential(x, y)
s.draw(ax=ax)
s.draw_field(*detector.centers, zorder=10, scale='log')

ax.legend(loc='upper right', fontsize='small')
ax.set_xlim(*(0.001293994677505457, 0.0022219773781759966))
ax.set_ylim(*(0.029440802941368933, 0.030296660409299757))

fig.show()
