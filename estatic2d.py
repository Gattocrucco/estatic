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
    
    @property
    def centers(self):
        return self._c + self._L / 2
    
    @property
    def areas(self):
        return np.prod(self._L, axis=0)
    
    @property
    def Ps(self):
        return self._Ps
    
    @Ps.setter
    def Ps(self, Ps):
        self._Ps = Ps
    
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
    
    def solve(self, zero_potential_at_infinity=True):
        shapes = np.array([len(conductor.lengths) for conductor in self.conductors])
        potentials = np.array([conductor.potential for conductor in self.conductors])
        B = np.empty(np.sum(shapes))
        idxs = np.cumsum(np.concatenate([[0], shapes]))
        for i in range(len(potentials)):
            B[idxs[i]:idxs[i+1]] = potentials[i]
        
        slopes = np.concatenate([conductor.slopes for conductor in self.conductors], axis=1)
        centers = np.concatenate([conductor.centers for conductor in self.conductors], axis=1)
        lengths = np.concatenate([conductor.lengths for conductor in self.conductors])
        
        assert len(slopes[0]) == len(centers[0]) == len(lengths) == len(B)
        
        # first dimension is fixed surface, second dimension is source
        a = np.sum(slopes ** 2, axis=0).reshape(1, -1)
        b = 2 * np.sum(slopes.reshape(2, 1, -1) * (centers.reshape(2, 1, -1) - centers.reshape(2, -1, 1)), axis=0)
        c = np.sum((centers.reshape(2, 1, -1) - centers.reshape(2, -1, 1)) ** 2, axis=0)
        
        assert a.shape == (1, len(B))
        assert b.shape == c.shape == (len(B), len(B))
        
        l = 1/2 * np.array([-lengths, lengths]).reshape(2, 1, -1)
        a = a.reshape(1, *a.shape)
        b = b.reshape(1, *b.shape)
        c = c.reshape(1, *c.shape)
        
        A = -1/2 * (1/a * np.sqrt(4*a*c - b**2) * np.arctan((2*a*l + b) / np.sqrt(4*a*c - b**2)) - 2*l + (b/(2*a) + l) * np.log(a*l**2 + b*l + c))
        
        assert A.shape == (2, len(B), len(B))
        
        A = A[1] - A[0]
        
        if zero_potential_at_infinity:
            A = np.concatenate([A, np.ones((len(B), 1)) * np.sum(lengths)], axis=1)
            A = np.concatenate([A, np.concatenate([lengths, [0]]).reshape(1, -1)], axis=0)
            B = np.concatenate([B, [0]])
        
        assert A.shape == (len(B), len(B))
        
        A /= 2 * np.pi * constants.epsilon_0
        
        solution = linalg.solve(A, B)
        
        if zero_potential_at_infinity:
            logr0 = solution[-1]
            sigmas = solution[:-1]
        else:
            logr0 = 0
            sigmas = solution
        
        for i in range(len(self.conductors)):
            self.conductors[i]._sigmas = sigmas[idxs[i]:idxs[i+1]]
        self._potential_offset = -np.sum(lengths) * logr0 / (2 * np.pi * constants.epsilon_0)
    
    @property
    def potential_offset(self):
        return self._potential_offset
