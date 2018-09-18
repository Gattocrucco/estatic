from matplotlib import pyplot as plt
import numpy as np
from scipy import constants, linalg

class Conductor(object):
    """docstring for Conductor"""
    def __init__(self, vertexes_x, vertexes_y, closed=True, potential=0, name='conductor'):
        vertexes_x = np.asarray(vertexes_x)
        vertexes_y = np.asarray(vertexes_y)
        
        assert len(vertexes_x.shape) == 1
        assert len(vertexes_y.shape) == 1
        assert len(vertexes_x) == len(vertexes_y)
        
        self.vertexes = np.array([vertexes_x, vertexes_y])
        
        factually_closed = np.all(self.vertexes[:,0] == self.vertexes[:,-1])
        assert not (factually_closed and not closed)
        if closed and not factually_closed:
            self.vertexes = np.concatenate([self.vertexes, self.vertexes[:,0:1]], axis=1)
            
        assert isinstance(name, str)
        
        self.closed = closed
        self.potential = potential
        self.name = name
        
        self._sigmas = None
    
    @property
    def lengths(self):
        return np.sqrt(np.sum(np.diff(self.vertexes, axis=1) ** 2, axis=0))
    
    @property
    def length(self):
        return np.sum(self.lengths)
    
    @property
    def centers(self):
        return (self.vertexes[:,1:] + self.vertexes[:,:-1]) / 2
    
    @property
    def slopes(self):
        return np.diff(self.vertexes, axis=1) / self.lengths
    
    @property
    def sigmas(self):
        return self._sigmas
    
    @property
    def charge_per_unit_length(self):
        return np.sum(self.sigmas * self.lengths)
        
    def draw(self, ax=None, **kw):
        if ax is None:
            ax = plt.gca()
        kwargs = dict(
            marker='.',
            markersize=4,
            label='{:s} V = {:.2g}'.format(self.name, self.potential)
        )
        kwargs.update(kw)
        return ax.plot(self.vertexes[0], self.vertexes[1], **kwargs)

class Circle(Conductor):
    """docstring for Segment"""
    def __init__(self, center=(0, 0), radius=1, segments=12, **kw):
        if isinstance(segments, int):
            angles = np.linspace(0, 2 * np.pi, segments + 1)[:-1]
        else:
            angles = np.asarray(segments)
            assert len(angles.shape) == 1
            assert len(angles) >= 2
        vertexes_x = center[0] + radius * np.cos(angles)
        vertexes_y = center[1] + radius * np.sin(angles)
        super(Circle, self).__init__(vertexes_x, vertexes_y, closed=True, **kw)

class Segment(Conductor):
    """docstring for Segment"""
    def __init__(self, endpoint_A=(0, 0), endpoint_B=(1, 1), segments=10, **kw):
        if isinstance(segments, int):
            steps = np.linspace(0, 1, segments + 1)
        else:
            steps = np.asarray(segments)
            assert len(steps.shape) == 1
            assert len(steps) >= 2
        vertexes_x = endpoint_A[0] + (endpoint_B[0] - endpoint_A[0]) * steps
        vertexes_y = endpoint_A[1] + (endpoint_B[1] - endpoint_A[1]) * steps
        super(Segment, self).__init__(vertexes_x, vertexes_y, closed=False, **kw)

class Dielectric(object):
    """list of rectangles"""
    def __init__(self, bottom_left_x, bottom_left_y, width, height, epsilon_rel=1, name='dielectric'):
        bottom_left_x = np.asarray(bottom_left_x)
        bottom_left_y = np.asarray(bottom_left_y)
        width = np.asarray(width)
        height = np.asarray(height)
        
        assert len(bottom_left_x.shape) == 1
        assert len(bottom_left_y.shape) == 1
        assert len(width.shape) == 1
        assert len(height.shape) == 1
        assert len(bottom_left_x) == len(bottom_left_y) == len(width) == len(height) > 0
        
        self._c = np.array([bottom_left_x, bottom_left_y])
        self._L = np.array([width, height])
        
        self._chi = float(epsilon_rel) - 1
        self._name = str(name)
        
        self._Ps = None
    
    @property
    def centers(self):
        return self._c + self._L / 2
    
    @property
    def areas(self):
        return np.prod(self._L, axis=0)
    
    @property
    def bottom_left(self):
        return self._c
    
    @property
    def sides(self):
        return self._L
    
    @property
    def polarizability(self):
        return self._chi
    
    @property
    def Ps(self):
        return self._Ps
    
    @property
    def polarization_per_unit_length(self):
        return np.sum(self.Ps * self.areas)

    def draw(self, ax=None, **kw):
        if ax is None:
            ax = plt.gca()
        kwargs = dict(
            label='{:s} $\\varepsilon$ = {:.2g}'.format(self._name, 1 + self._chi)
        )
        kwargs.update(kw)
        
        l = self._c
        r = self._c + self._L
        
        x = np.array([l[0], l[0], r[0], r[0], l[0]])
        y = np.array([l[1], r[1], r[1], l[1], l[1]])
        
        lines = []
        line, = ax.plot(x[:,0], y[:,0], **kwargs)
        kwargs.update(color=line.get_color(), label=None)
        lines.append(line)
        for i in range(1, x.shape[1]):
            lines.append(ax.plot(x[:,i], y[:,i], **kwargs)[0])
        return lines

class Rectangle(Dielectric):
    """docstring for Rectangle"""
    def __init__(self, bottom_left=(0, 0), sides=(1, 1), segments=(10, 10), **kw):
        bottom_left_x = bottom_left[0] + sides[0] * np.arange(segments[0]) / segments[0]
        bottom_left_y = bottom_left[1] + sides[1] * np.arange(segments[1]) / segments[1]

        base = np.ones((segments[0], segments[1]))
        bottom_left_x = base * bottom_left_x.reshape(-1, 1)
        bottom_left_y = base * bottom_left_y.reshape(1, -1)

        width = base * sides[0] / segments[0]
        height = base * sides[1] / segments[1]
        
        super(Rectangle, self).__init__(bottom_left_x.flatten(), bottom_left_y.flatten(), width.flatten(), height.flatten(), **kw)        

class ConductorSet(object):
    """docstring for ConductorSet"""
    def __init__(self, *conductors_and_dielectrics):
        self._conductors = []
        self._dielectrics = []
        for obj in conductors_and_dielectrics:
            if isinstance(obj, Conductor):
                self._conductors.append(obj)
            elif isinstance(obj, Dielectric):
                self._dielectrics.append(obj)
            else:
                raise ValueError('unrecognized object')
        self._conductors = tuple(self._conductors)
        self._dielectrics = tuple(self._dielectrics)
        assert len(self._conductors) > 0
    
    @property
    def conductors(self):
        return self._conductors
    
    @property
    def dielectrics(self):
        return self._dielectrics
    
    def draw(self, *args, **kw):
        rt = []
        for conductor in self.conductors:
            rt += conductor.draw(*args, **kw)
        for dielectric in self.dielectrics:
            rt += dielectric.draw(*args, **kw)
        return rt
    
    def solve(self, zero_potential_at_infinity=True, use_dielectrics=True):
        epsilon_0 = 1 #constants.epsilon_0
        
        # extract conductor properties
        cond_shapes = np.array([len(conductor.lengths) for conductor in self.conductors])
        cond_potentials = np.concatenate([
            np.ones(shape) * conductor.potential
            for shape, conductor in zip(cond_shapes, self.conductors)
        ])
        cond_slopes = np.concatenate([conductor.slopes for conductor in self.conductors], axis=1)
        cond_centers = np.concatenate([conductor.centers for conductor in self.conductors], axis=1)
        cond_lengths = np.concatenate([conductor.lengths for conductor in self.conductors])
        N_cond = len(cond_potentials)
        
        assert len(cond_slopes[0]) == len(cond_centers[0]) == len(cond_lengths) == len(cond_potentials)

        # extract dielectric properties
        if len(self.dielectrics) == 0:
            use_dielectrics = False
        if use_dielectrics:
            diel_shapes = np.array([len(dielectric.areas) for dielectric in self.dielectrics])
            diel_chis = np.concatenate([
                np.ones(shape) * dielectric.polarizability
                for shape, dielectric in zip(diel_shapes, self.dielectrics)
            ])
            diel_centers = np.concatenate([dielectric.centers for dielectric in self.dielectrics], axis=1)
            diel_bottom_left = np.concatenate([dielectric.bottom_left for dielectric in self.dielectrics], axis=1)
            diel_sides = np.concatenate([dielectric.sides for dielectric in self.dielectrics], axis=1)
            N_diel = len(diel_chis)
            
            assert len(diel_chis) == len(diel_centers[0]) == len(diel_bottom_left[0]) == len(diel_sides[0])
        else:
            N_diel = 0
        
        # linear system to solve is Ax=B
        # layout of the equations:
        # *––––––––––––*––––––––––––*   *–––––––––––*
        # | cond<-cond | cond<-diel |   | potential |
        # |––––––––––––|––––––––––––|   |–––––––––––|
        # | diel<-cond | diel<-diel | = |     0     |
        # |––––––––––––|––––––––––––|   |–––––––––––|
        # | sum charge |            |   |     0     |
        # *––––––––––––*––––––––––––*   *–––––––––––*
        # 
        # arrangement of unknowns:
        # [*sigma, *P_x, *P_y, logr0]
        
        # construct B
        B_potential = cond_potentials
        B_polarization = np.zeros(2 * N_diel if use_dielectrics else 0)
        B_boundary_conditions = np.zeros(1 if zero_potential_at_infinity else 0)
        B = np.concatenate([
            B_potential,
            B_polarization,
            B_boundary_conditions
        ])
        
        # construct A_cc
        
        # careful: l, a used also in A_dc
        a = np.sum(cond_slopes ** 2, axis=0).reshape(1, -1)
        b = 2 * np.sum(cond_slopes.reshape(2, 1, -1) * (cond_centers.reshape(2, 1, -1) - cond_centers.reshape(2, -1, 1)), axis=0)
        c = np.sum((cond_centers.reshape(2, 1, -1) - cond_centers.reshape(2, -1, 1)) ** 2, axis=0)
        
        assert b.shape == c.shape == 2 * (N_cond,)
        assert a.shape == (1, N_cond)
        
        l = 1/2 * np.array([-cond_lengths, cond_lengths]).reshape(2, 1, -1)
        a = a.reshape(1, *a.shape)
        b = b.reshape(1, *b.shape)
        c = c.reshape(1, *c.shape)
        
        A_cc = -1/2 * (1/a * np.sqrt(4*a*c - b**2) * np.arctan((2*a*l + b) / np.sqrt(4*a*c - b**2)) - 2*l + (b/(2*a) + l) * np.log(a*l**2 + b*l + c))
        
        assert A_cc.shape == (2, N_cond, N_cond)
        
        A_cc = (A_cc[1] - A_cc[0]) / (2 * np.pi * epsilon_0)
        
        if use_dielectrics:
            # construct A_dc
        
            b = 2 * np.sum(cond_slopes.reshape(2, 1, -1) * (cond_centers.reshape(2, 1, -1) - diel_centers.reshape(2, -1, 1)), axis=0)
            c = np.sum((cond_centers.reshape(2, 1, -1) - diel_centers.reshape(2, -1, 1)) ** 2, axis=0)
            d = -cond_slopes.reshape(2, 1, -1)
            e = diel_centers.reshape(2, -1, 1) - cond_centers.reshape(2, 1, -1)
        
            assert b.shape == c.shape == (N_diel, N_cond)
            assert d.shape == (2, 1, N_cond)
            assert e.shape == (2, N_diel, N_cond)
        
            l = l.reshape(2, 1, 1, -1)
            a = a.reshape(1, *a.shape)
            b = b.reshape(1, 1, *b.shape)
            c = c.reshape(1, 1, *c.shape)
            d = d.reshape(1, *d.shape)
            e = e.reshape(1, *e.shape)
        
            A_dc = d/(2*a) * np.log(a*l**2 + b*l + c) + (2*e - b*d/a) / np.sqrt(4*a*c - b**2) * np.arctan((2*a*l + b) / np.sqrt(4*a*c - b**2))
        
            assert A_dc.shape == (2, 2, N_diel, N_cond)
        
            A_dc = (A_dc[1] - A_dc[0]) * diel_chis.reshape(1, -1, 1) / (2 * np.pi)
            A_dc = A_dc.reshape(2 * N_diel, N_cond)
        
            # construct A_cd
        
            uv = np.array([
                diel_bottom_left.reshape(2, -1, 1) - cond_centers.reshape(2, 1, -1),
                diel_bottom_left.reshape(2, -1, 1) - cond_centers.reshape(2, 1, -1) + diel_sides.reshape(2, -1, 1)
            ])
            u = uv[:,0]
            v = uv[:,1]
        
            f = lambda u, v: -1/2 * (v * np.log(u**2 + v**2) + 2*u * np.arctan(v/u) - 2*v)
            delta = lambda f, u, v: f(u[1], v[1]) - f(u[1], v[0]) - f(u[0], v[1]) + f(u[0], v[0])
            A_cd_x = delta(f, u, v)
            A_cd_y = delta(f, v, u)
        
            A_cd = np.array([A_cd_x, A_cd_y])
            A_cd /= 2 * np.pi * epsilon_0
            A_cd = A_cd.reshape(2 * N_diel, N_cond).transpose()
            # We did the transposition at the end because, if done before, flattening x and y would have been more complicated.
        
            # construct A_dd
        
            uv = np.array([
                diel_bottom_left.reshape(2, -1, 1) - diel_centers.reshape(2, 1, -1),
                diel_bottom_left.reshape(2, -1, 1) - diel_centers.reshape(2, 1, -1) + diel_sides.reshape(2, -1, 1)
            ])
            u = uv[:,0]
            v = uv[:,1]
        
            f = lambda u, v: -np.arctan(v / u)
            g = lambda u, v: -1/2 * np.log(u**2 + v**2)
            A_dd_Px_x = delta(f, u, v) * diel_chis.reshape(1, -1)
            A_dd_Py_x = delta(g, u, v) * diel_chis.reshape(1, -1)
            A_dd_Px_y = A_dd_Py_x # because symmetric in u, v
            A_dd_Py_y = delta(f, v, u) * diel_chis.reshape(1, -1)
        
            A_dd_Px = np.concatenate([A_dd_Px_x, A_dd_Px_y], axis=1)
            A_dd_Py = np.concatenate([A_dd_Py_x, A_dd_Py_y], axis=1)
            A_dd = np.concatenate([A_dd_Px, A_dd_Py], axis=0).transpose()
            A_dd /= 2 * np.pi
            np.fill_diagonal(A_dd, np.diagonal(A_dd) - 1)
        
            assert A_dd.shape == (2 * N_diel, 2 * N_diel)
        else:
            A_cd = np.zeros((N_cond, 0))
            A_dc = np.zeros((0, N_cond))
            A_dd = np.zeros((0, 0))
        
        if zero_potential_at_infinity:
            # construct A_offset
            A_offset = np.ones((N_cond, 1)) * np.sum(cond_lengths) / (2 * np.pi * epsilon_0)
        
            # construct A_charge
            A_charge = cond_lengths.reshape(1, -1)
            
            # various zeroes of A
            A_offset_diel = np.zeros((2 * N_diel, 1))
            A_charge_diel = np.zeros((1, 2 * N_diel))
            A_charge_offset = np.zeros((1, 1))
        else:
            A_offset = np.zeros((N_cond, 0))
            A_charge = np.zeros((0, N_cond))
            A_offset_diel = np.zeros((2 * N_diel, 0))
            A_charge_diel = np.zeros((0, 2 * N_diel))
            A_charge_offset = np.zeros((0, 0))
        
        A_cc_cd_offset = np.concatenate([A_cc, A_cd, A_offset], axis=1)
        A_dc_dd = np.concatenate([A_dc, A_dd, A_offset_diel], axis=1)
        A_charge = np.concatenate([A_charge, A_charge_diel, A_charge_offset], axis=1)
        A = np.concatenate([A_cc_cd_offset, A_dc_dd, A_charge], axis=0)
        
        assert A.shape == (len(B), len(B))
                
        solution = linalg.solve(A, B)
        
        sigmas = solution[:N_cond] * constants.epsilon_0 / epsilon_0
        if use_dielectrics:
            Ps = np.array([
                solution[N_cond:N_cond + N_diel],
                solution[N_cond + N_diel:N_cond + 2 * N_diel]
            ]) * constants.epsilon_0 / epsilon_0
        if zero_potential_at_infinity:
            logr0 = solution[-1]
        else:
            logr0 = 0
        
        idxs = np.cumsum(np.concatenate([[0], cond_shapes]))
        for i in range(len(self.conductors)):
            self.conductors[i]._sigmas = sigmas[idxs[i]:idxs[i+1]]
       
        if use_dielectrics:
            idxs = np.cumsum(np.concatenate([[0], diel_shapes]))
            for i in range(len(self.dielectrics)):
                self.dielectrics[i]._Ps = Ps[idxs[i]:idxs[i+1]]
        
        self._potential_offset = -np.sum(cond_lengths) * logr0 / (2 * np.pi * epsilon_0)
    
    @property
    def potential_offset(self):
        return self._potential_offset
