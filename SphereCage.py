from __future__ import division
from AbstractCage import AbstractCage
import numpy as np
from Quaternion import QuaternionBetween

class SphereCage(AbstractCage):

    def __init__(self, radius):

        super(SphereCage, self).__init__()
        sa = 4 * np.pi * radius * radius
        area = .25 * np.pi
        length = int(sa/area)
        points = np.multiply(self.unit_shpere(length), radius)
        for ind, point in enumerate(points):
            self.positions.append(point)
            self.masses.append(1)
            self.types.append('N')
            self.charges.append(0)
            self.images.append([0,0,0])

        pqr = "parse_pH6.5_SingleCage2_noraft.pqr"
        pqr_pos, pqr_charges = self.read_pqr(pqr)
        quat = QuaternionBetween(np.divide([1, 1, 1], np.sqrt(3)), [1, 0, 0])
        self.align(quat)
        self.assign_charges_vector(pqr_pos, pqr_charges)
        self.align(quat)

        self.num_rigid = len(points)
        self.num_connected = len(points)
        self.moment = self.calculate_inertia_tensor()


    def unit_shpere(self, n):

        points = []
        offset = 2. / n
        increment = np.pi * (3 - np.sqrt(5))

        for i in range(n):
            y = ((i * offset) - 1) + offset/2
            r = np.sqrt(1 - pow(y,2))
            phi = i * increment
            x = np.cos(phi) * r
            z = np.sin(phi) * r
            points.append([x, y, z])
        return points