import numpy as np
from numpy import linalg as la
import copy as cp
from AbstractCage import AbstractCage

class DecoratedCage(object):

    def __init__(self, basic_cage):
        self.my_type = "Decorated"
        if basic_cage.my_type == "Abstract":
            self.cage = cp.deepcopy(basic_cage)
            self.bonds = []
            self.bond_names = []
            self.num_connected = len(basic_cage.num_rigid)
            self.angles = []
            self.angle_names = []
            #self.graft_sites = []
        elif basic_cage.my_type == "Decorated":
            self.cage = cp.deepcopy(basic_cage.cage)
            self.bonds = cp.deepcopy(basic_cage.bonds)
            self.bond_names = cp.deepcopy(basic_cage.bond_names)
            self.num_connected = cp.deepcopy(basic_cage.num_connected)

    def add_linker(self, index):

        pos = self.cage.positions[index]
        pos = np.add(pos, np.mutiply(la.norm(pos), .5))
        self.cage.positions.append(pos)
        self.cage.types.append("L")
        self.cage.masses.append(1)
        self.cage.charges.append(0)
        self.cage.images.append([0, 0, 0])
        self.bonds.append([index, self.num_connected])
        self.bond_names.append("linker")
        self.num_connected += 1

    def add_graft_site(self, index):
        if self.cage.num_rigid != self.num_connected:
            raise PermissionError("Cannot add graft sites after linkers")

        pos = self.cage.positions[index]
        pos = np.add(pos, np.mutiply(la.norm(pos), .5))
        self.cage.positions.append(pos)
        self.cage.types.append("G")
        self.cage.masses.append(1)
        self.cage.charges.append(0)
        self.cage.images.append([0, 0, 0])
        self.cage.num_rigid += 1


    def get_positions(self):
        return self.cage.positions

    def set_index(self, index):
        self.cage.set_index(index)

    def get_index(self):
        return self.cage.index

    def get_bonds(self):
        return self.bonds

    def get_bond_names(self):
        return self.bond_names

    def get_angles(self):
        return self.angles

    def get_angle_names(self):
        return self.angle_names

class NeighborLinkerCage(DecoratedCage):

    def __init__(self, basic_cage, points='fcc'):

        super(NeighborLinkerCage).__init__(basic_cage)
        if points == 'fcc':
            neighbor_points = [[1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
                      [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
                      [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0]]
        else:
            raise IOError("Dont have the points for " + str(points))
        indexes = [0 for _ in range(12)]
        dots = [0 for _ in range(12)]

        for ind in range(self.cage.num_rigid):
            for ind2, point in enumerate(neighbor_points):
                dot = np.dot(point, self.cage.positions[ind])
                if dot > dots[ind2]:
                    indexes[ind2] = ind
        for index in indexes:
            self.add_linker(index)


class SixFoldGraftedCage(DecoratedCage):

    def __init__(self, basic_cage):

        super(SixFoldGraftedCage).__init__(basic_cage)
        neighbor_points = [[0, 0, 1], [0, 0, -1], [1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0]]
        dots = [0 for _ in range(6)]
        indexes = [0 for _ in range(6)]
        for ind in range(self.cage.num_rigid):
            for ind2, point in enumerate(neighbor_points):
                dot = np.dot(point, self.cage.positions[ind])
                if dot > dots[ind2]:
                    indexes[ind2] = ind
        for index in indexes:
            self.add_graft_site(index)