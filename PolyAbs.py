from __future__ import division
import numpy as np
from numpy import linalg as la
from Quaternion import QuaternionBetween
import math
import copy as cp


class PolyAbs(object):

    def __init__(self, sequence, index=-1, seperation=4, with_ion=False):


        self.position = []
        self.type = []
        self.bonds = []
        self.bond_names = []
        self.angles = []
        self.angle_names = []
        self.charge = []
        self.length = 0
        self.rigid_count = 0
        self.mass = []
        self.index = index
        self.sequence = sequence
        self.monomer_indexes = []
        self.num_beads = 0
        self.body = []
        self.seperation = seperation
        self.image = []
        self.build_chain(sequence, with_ion=with_ion)

    def build_chain(self, sequence, with_ion=False):
        print("I'm an abstract class and I ain't building shit")

    def align(self, vec):
        q = QuaternionBetween(self.chain_vector(), vec)
        for x in range(len(self.position)):
            self.position[x] = q.orient(self.position[x])

    def align_to_q(self, q):
        for x in range(len(self.position)):
            self.position[x] = q.orient(self.position[x])

    def shift(self, vector):

        for ind, site in enumerate(self.position):
            self.position[ind] = np.add(site, vector)

    def chain_vector(self):

        return np.subtract(self.position[-1], self.position[0])

    def rigid_center_of_mass(self):

        mass = self.mass

        mass_array = np.array(mass)
        position_array = np.array(self.position)
        return np.sum(self.pos_x_mass(position_array, mass_array), axis=0) / np.sum(mass_array)

    def pos_x_mass(self, position_array, mass_array):

        y = np.zeros_like(position_array)
        for ind in range(len(mass_array)):
            y[ind][0] = position_array[ind][0] * mass_array[ind]
            y[ind][1] = position_array[ind][1] * mass_array[ind]
            y[ind][2] = position_array[ind][2] * mass_array[ind]
        return y

    def moment_inertia(self):
        mass = self.mass

        mass_array = np.array(mass)
        position_array = np.array(self.position)

        cen = self.center_of_mass_arrays(position_array, mass_array)
        position_array = np.subtract(position_array, cen)
        return self.sum_over_xyz(self.pos_x_mass(self.pos_squared(position_array), mass_array))

    def pos_squared(self, position_array):

        y = np.zeros_like(position_array)

        for ind in range(len(position_array)):
            y[ind][0] = position_array[ind][0] * position_array[ind][0]
            y[ind][1] = position_array[ind][1] * position_array[ind][1]
            y[ind][2] = position_array[ind][2] * position_array[ind][2]
        return y

    def sum_over_xyz(self, array):

        final = np.array([0, 0, 0])

        for list in array:
            final[0] += list[0]
            final[1] += list[1]
            final[2] += list[2]
        return final

    def center_of_mass_arrays(self, position_array, mass_array):
        return np.sum(self.pos_x_mass(position_array, mass_array), axis=0) / np.sum(mass_array)

    def max_radius(self):

        cen = self.rigid_center_of_mass()
        max = 0
        for pos in self.position:
            dist = la.norm(np.subtract(pos, cen))
            if dist > max:
                max = dist
        return max

    def spiral_points(self,n, arc=.5, separation=4):
        """generate points on an Archimedes' spiral
        with `arc` giving the length of arc between two points
        and `separation` giving the distance between consecutive
        turnings
        - approximate arc length with circle arc at given distance
        - use a spiral equation r = b * phi
        """

        def p2c(r, phi):
            """polar to cartesian
            """
            return [r * math.cos(phi), r * math.sin(phi), 0]

        # yield a point at origin
        points=  [[0,0,0]]

        # initialize the next point in the required distance
        r = arc
        b = separation / (2 * math.pi)
        # find the first phi to satisfy distance of `arc` to the second point
        phi = float(r) / b
        count = 0
        while count < n - 1:
            points.append(p2c(r, phi))
            # advance the variables
            # calculate phi that will give desired arc length at current radius
            # (approximating with circle)
            phi += float(arc) / r
            r = b * phi
            count += 1
        return points

    def linear_points(self, number, spacing):

        return [[x * spacing, 0, 0] for x in range(number)]

    def geometric_center(self):

        return np.average(self.position, axis=0)

    def enforce_cubic_bc(self, box_length):

        self.image = [[0, 0, 0] for _ in range(len(self.position))]
        half = box_length / 2
        for ind1, position in enumerate(self.position):
            if not self.is_ion(ind1):
                for ind2 in range(0, 3):
                    if position[ind2] > box_length + half:
                        raise ValueError("Polymer bead is greater than a box length outside")
                    elif position[ind2] > half:
                        self.position[ind1][ind2] -= box_length
                        self.image[ind1][ind2] += 1
                    elif position[ind2] < -1 * (box_length + half):
                        raise ValueError("Polymer bead is greater than a box length outside")
                    elif position[ind2] < -1 * half:
                        self.position[ind1][ind2] += box_length
                        self.image[ind1][ind2] -= 1

    @staticmethod
    def sdot(v1, v2):
        return np.sum([v1[i] * v2[i] for i in range(3)])

    def enforce_generic_bc(self, cell_matrix):

        changed = False
        v = [[] for _ in range(3)]

        v[0] = cell_matrix[:, 0]
        v[1] = cell_matrix[:, 1]
        v[2] = cell_matrix[:, 2]
        L = np.diag(cell_matrix)
        point_top = np.divide([np.sum(cell_matrix[i]) for i in range(3)], 2)
        point_bottom = np.multiply(point_top, -1)
        #print("point_top", point_top)
        #print("point_bottom",point_bottom)
        plane0 = np.cross(v[0], v[1])
        plane0 = np.multiply(plane0, la.norm(cell_matrix[2])/la.norm(plane0))
        d00 = self.sdot(plane0, point_bottom)
        d01 = self.sdot(plane0, point_top)

        plane1 = np.cross(-v[0], v[2])
        plane1 = np.multiply(plane1, la.norm(cell_matrix[1]) / la.norm(plane1))
        d10 = self.sdot(plane1, point_bottom)
        d11 = self.sdot(plane1, point_top)

        plane2 = np.cross(v[1], v[2])
        plane2 = np.multiply(plane2, la.norm(cell_matrix[0]) / la.norm(plane2))
        d20 = self.sdot(plane2, point_bottom)
        d21 = self.sdot(plane2, point_top)

        #print(plane0, d00, d01)
        #print(plane1, d10, d11)
        #print(plane2, d20, d21)
        for ind in range(len(self.position)):
            original = cp.deepcopy(self.position[ind])
            d0 = self.sdot(plane0, self.position[ind])
            if np.subtract(d0, la.norm(plane0)**2) > d01:
                raise ValueError("bead is greater than a box length outside", self.position[ind], cell_matrix)
            elif d0 > d01:
                self.position[ind] = np.subtract(self.position[ind], v[2])
                self.image[ind][2] += 1
                changed = True
            elif np.add(d0, la.norm(plane0)**2) < d00:
                raise ValueError("bead is greater than a box length outside", self.position[ind], cell_matrix)
            elif d0 < d00:
                self.position[ind] = np.add(self.position[ind], v[2])
                self.image[ind][2] -= 1
                changed = True
            d1 = self.sdot(plane1, self.position[ind])
            if np.subtract(d1, la.norm(plane1)**2) > d11:
                print("original", original, "image", self.image[ind])
                raise ValueError("bead is greater than a box length outside", self.position[ind], cell_matrix)
            elif d1 > d11:
                self.position[ind] = np.subtract(self.position[ind], v[1])
                self.image[ind][1] += 1
                changed = True
            elif np.add(d1, la.norm(plane1)**2) < d10:
                raise ValueError("bead is greater than a box length outside", self.position[ind], cell_matrix)
            elif d1 < d10:
                self.position[ind] = np.add(self.position[ind], v[1])
                self.image[ind][1] -= 1
                changed = True

            d2 = self.sdot(plane2, self.position[ind])
            if np.subtract(d2, la.norm(plane2)**2) > d21:
                raise ValueError("Bead is greater than a box length outside", self.position[ind], cell_matrix)
            elif d2 > d21:
                self.position[ind] = np.subtract(self.position[ind], v[0])
                self.image[ind][0] += 1
                changed = True
            elif np.add(d2, la.norm(plane2)**2) < d20:
                raise ValueError("Bead is greater than a box length outside", self.position[ind], cell_matrix)
            elif d2 < d20:
                self.position[ind] = np.add(self.position[ind], v[0])
                self.image[ind][0] -= 1
                changed = True

        return changed

    def check_length(self, position, box_length):
        image = 0
        pos = cp.deepcopy(position)
        half = box_length / 2
        if pos > box_length + half:
            raise ValueError("bead is greater than a box length outside")
        elif pos > half:
            pos -= box_length
            image += 1
        elif pos < -1 * (box_length + half):
            raise ValueError("bead is greater than a box length outside")
        elif pos < -1 * half:
            pos += box_length
            image -= 1
        return pos, image

    def is_ion(self, ind):

        return self.type[ind][-1] == "i"