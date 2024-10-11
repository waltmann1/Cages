from __future__ import division
import numpy as np
from numpy import linalg as la
import copy as cp


class AbstractCage(object):

    def __init__(self):

        self.positions = []
        self.types = []
        self.masses = []
        self.charges = []
        self.orientation = [1, 0, 0, 0]
        self.images = []
        self.center = [0, 0, 0]
        self.num_rigid = 0
        self.index = 0
        self.bonds = []
        self.bond_names = []
        self.num_connected = 0
        self.angles = []
        self.angle_names = []
        self.graft_sites = []
        self.grafted = []
        self.moment = []

    def shift(self, vector):

        for ind, site in enumerate(self.positions):
            self.positions[ind] = np.add(site, vector)

        self.center = np.add(self.center, vector)

    def align(self, quat):

        for ind, site in enumerate(self.positions):
            self.positions[ind] = quat.orient(site)

        self.orientation = quat.q

        return [quat.q[3], quat.q[0], quat.q[1], quat.q[2]]

    def max_radius(self):

        cen = self.rigid_center_of_mass()
        max = 0
        for ind, pos in enumerate(self.positions):
            dist = la.norm(np.subtract(pos, cen))
            if dist > max:
                max = dist

        return max

    def rigid_max_radius(self):
        cen = self.center
        max = 0
        for pos in self.positions[:self.num_rigid]:
            dist = la.norm(np.subtract(cen, pos))
            if dist > max:
                max = dist
        return max

    def calculate_inertia_tensor(self):

        tensor = np.zeros((3, 3))
        temp = self.center
        self.center = self.rigid_center_of_mass()
        position_array = np.subtract(self.positions[:self.num_rigid], self.center)
        mass_array = self.masses[:self.num_rigid]
        for idx in range(len(mass_array)):
            tensor[0][0] += (np.square(position_array[idx][1]) + np.square(position_array[idx][2])) * mass_array[
                idx]
            tensor[1][1] += (np.square(position_array[idx][2]) + np.square(position_array[idx][0])) * mass_array[
                idx]

            tensor[2][2] += (np.square(position_array[idx][0]) + np.square(position_array[idx][1])) * mass_array[
                idx]
            tensor[0][1] -= position_array[idx][0] * position_array[idx][1] * mass_array[idx]
            tensor[0][2] -= position_array[idx][0] * position_array[idx][2] * mass_array[idx]
            tensor[1][2] -= position_array[idx][1] * position_array[idx][2] * mass_array[idx]
            tensor[1][0] = tensor[0][1]
            tensor[2][0] = tensor[0][2]
            tensor[2][1] = tensor[1][2]

            values, vectors = la.eig(tensor)
        self.center = temp
        return values

    def rigid_center_of_mass(self):


        mass = self.masses[:self.num_rigid]
        mass_array = np.array(mass)
        position_array = np.array(self.positions[:self.num_rigid])
        array = [0, 0, 0]

        for i in range(self.num_rigid):
            for x in range(3):
                array[x] += mass_array[i] * position_array[i][x]

        return array / np.sum(mass_array)

    def center_of_mass(self):

        mass = self.masses
        mass_array = np.array(mass)
        position_array = np.array(self.positions)
        array = [0, 0, 0]

        for i in range(self.num_rigid):
            for x in range(3):
                array[x] += mass_array[i] * position_array[i][x]

        return array / np.sum(mass_array)

    def rigid_mass(self):
        return np.sum(self.masses[:self.num_rigid])

    def enforce_cubic_bc(self, box_length):

        #self.image = [[0, 0, 0] for _ in range(len(self.positions))]
        half = box_length / 2
        changed = False
        for ind1, position in enumerate(self.positions):
                for ind2 in range(0, 3):
                    if position[ind2] > box_length + half:
                        raise ValueError("bead is greater than a box length outside")
                    elif position[ind2] > half:
                        self.positions[ind1][ind2] -= box_length
                        self.image[ind1][ind2] += 1
                        changed = True
                        # print("fixed", self.position[ind1], self.image[ind1])
                    elif position[ind2] < -1 * (box_length + half):
                        raise ValueError("bead is greater than a box length outside")
                    elif position[ind2] < -1 * half:
                        self.positions[ind1][ind2] += box_length
                        self.image[ind1][ind2] -= 1
                        changed = True
        return changed

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
        for ind in range(len(self.positions)):
            d0 = self.sdot(plane0, self.positions[ind])
            if np.subtract(d0, la.norm(plane0)**2) > d01:
                raise ValueError("bead is greater than a box length outside", self.positions[ind], cell_matrix)
            elif d0 > d01:
                self.positions[ind] = np.subtract(self.positions[ind], v[2])
                self.images[ind][2] += 1
                changed = True
            elif np.add(d0, la.norm(plane0)**2) < d00:
                raise ValueError("bead is greater than a box length outside", self.positions[ind], cell_matrix)
            elif d0 < d00:
                self.positions[ind] = np.add(self.positions[ind], v[2])
                self.images[ind][2] -= 1
                changed = True
            d1 = self.sdot(plane1, self.positions[ind])
            if np.subtract(d1, la.norm(plane1)**2) > d11:
                raise ValueError("bead is greater than a box length outside", self.positions[ind], cell_matrix)
            elif d1 > d11:
                self.positions[ind] = np.subtract(self.positions[ind], v[1])
                self.images[ind][1] += 1
                changed = True
            elif np.add(d1, la.norm(plane1)**2) < d10:
                raise ValueError("bead is greater than a box length outside", self.positions[ind], cell_matrix)
            elif d1 < d10:
                self.positions[ind] = np.add(self.positions[ind], v[1])
                self.images[ind][1] -= 1
                changed = True

            d2 = self.sdot(plane2, self.positions[ind])
            if np.subtract(d2, la.norm(plane2)**2) > d21:
                raise ValueError("Bead is greater than a box length outside", self.positions[ind], cell_matrix)
            elif d2 > d21:
                self.positions[ind] = np.subtract(self.positions[ind], v[0])
                self.images[ind][0] += 1
                changed = True
            elif np.add(d2, la.norm(plane2)**2) < d20:
                raise ValueError("Bead is greater than a box length outside", self.positions[ind], cell_matrix)
            elif d2 < d20:
                self.positions[ind] = np.add(self.positions[ind], v[0])
                self.images[ind][0] -= 1
                changed = True

        return changed

    def add_linker(self, index):

        pos = self.positions[index]
        vec = np.multiply(np.divide(pos, la.norm(pos)), 1.0)
        pos = np.add(pos, vec)
        self.positions.append(pos)
        self.types.append("N")
        self.masses.append(1)
        self.charges.append(0)
        self.images.append([0, 0, 0])
        self.bonds.append([index, self.num_connected])
        self.bond_names.append("linker")

        pos = self.positions[-1]
        vec = np.multiply(np.divide(pos, la.norm(pos)), 0.5)
        pos = np.add(pos, vec)
        self.positions.append(pos)
        self.types.append("L")
        self.masses.append(1)
        self.charges.append(0)
        self.images.append([0, 0, 0])
        self.bonds.append([self.num_connected, self.num_connected + 1])
        self.bond_names.append("linker2")
        self.angles.append([index, self.num_connected, self.num_connected + 1])
        self.angle_names.append("linker")

        self.num_connected += 2

    def add_graft_site(self, index):
        if self.num_rigid != self.num_connected:
            raise PermissionError("Cannot add graft sites after linkers")

        pos = self.positions[index]
        vec = np.multiply(np.divide(pos, la.norm(pos)), .5)
        pos = np.add(pos, vec)
        self.positions.append(pos)
        self.types.append("G")
        self.masses.append(1)
        self.charges.append(0)
        self.images.append([0, 0, 0])
        self.graft_sites.append(self.num_rigid)
        self.num_rigid += 1
        self.num_connected += 1
        self.grafted.append(None)
        self.moment = self.calculate_inertia_tensor()

    def add_neighbor_linkers(self, points="fcc"):

        if points == 'fcc':
            neighbor_points = [[1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
                      [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
                      [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1]]
        else:
            raise IOError("Dont have the points for " + str(points))
        indexes = [0 for _ in range(12)]
        dots = [0 for _ in range(12)]

        for ind2, point in enumerate(neighbor_points):
            for ind in range(self.num_rigid):
                dot = np.sum(point[i] * self.positions[ind][i] for i in range(3)) / la.norm(self.positions[ind] / la.norm(point))
                if dot > dots[ind2]:
                    indexes[ind2] = ind
                    dots[ind2] = dot
            #print("link site", point, self.positions[indexes[ind2]], dots[ind2])
        for index in indexes:
            self.add_linker(index)

    def add_six_fold_graft_sites(self):

        neighbor_points = [[0, 0, 1], [0, 0, -1], [1, 0, 0], [-1, 0, 0], [0, -1, 0], [0, 1, 0]]
        dots = [0 for _ in range(6)]
        indexes = [0 for _ in range(6)]
        for ind2, point in enumerate(neighbor_points):
            for ind in range(self.num_rigid):
                dot = np.sum(point[i] * self.positions[ind][i] for i in range(3))
                if dot > dots[ind2]:
                    indexes[ind2] = ind
                    dots[ind2] = dot
            #print("graft site", point, self.positions[indexes[ind2]], dots[ind2])
        for index in indexes:
            self.add_graft_site(index)

    def graft(self, polymer, graft_index):

        self.grafted[graft_index] = polymer
        graft_pos = self.positions[self.graft_sites[graft_index]]
        poly_positions = np.add(polymer.position, graft_pos)
        self.positions.extend(poly_positions)
        self.types.extend(polymer.type)
        self.masses.extend(polymer.mass)
        self.charges.extend(polymer.charge)
        self.images.extend(polymer.image)
        self.bond_names.extend(polymer.bond_names)
        self.bonds.extend(np.add(polymer.bonds, self.num_connected))
        self.angle_names.extend(polymer.angle_names)
        self.angles.extend(np.add(polymer.angles, self.num_connected))
        self.num_connected += len(polymer.position)

    def dump_xyz(self, title):

        num = len(self.positions)

        f = open(title + ".xyz", "w")
        f.write(str(num))
        f.write("\n\n")
        count = 0
        mon_count = 0


        for mon in range(num):
            count += 1
            mon_count += 1
            s = "%5s%8.3f%8.3f%8.3f\n" % (
                    self.types[mon], self.positions[mon][0], self.positions[mon][1], self.positions[mon][2])
            f.write(s)
            mon_count = 0
        f.close()

    def dump_qxyz(self, title):

        num = len(self.positions)

        f = open(title + ".qxyz", "w")
        f.write(str(num))
        f.write("\n\n")
        count = 0
        mon_count = 0


        for mon in range(num):
            count += 1
            mon_count += 1
            s = "%5s%8.3f%8.3f%8.3f\n" % (
                    self.charges[mon], self.positions[mon][0], self.positions[mon][1], self.positions[mon][2])
            f.write(s)
            mon_count = 0
        f.close()

    def read_pqr(self, filename):

        f = open(filename, 'r')
        positions = []
        charges = []
        data = f.readlines()
        for line in data:
            s = line.split()
            if s[0] == "ATOM":
                positions.append([float(s[5]) / 10, float(s[6]) / 10, float(s[7]) / 10])
                charges.append(float(s[8]))
        center = np.average(positions, axis=0)
        for ind, pos in enumerate(positions):
            positions[ind] = np.subtract(pos, center)
        return positions, charges

    def assign_charges_position(self, pqr_positions, pqr_charges):

        for i in range(len(pqr_positions)):
            if pqr_charges[i] != 0:
                min = 1000
                index = 0
                for j in range(len(self.positions)):
                    dist = la.norm(np.subtract(self.positions[j], pqr_positions[i]))
                    if dist < min:
                        min = dist
                        index = j
                self.charges[index] += pqr_charges[i]
                print(pqr_charges[i], min)


    def assign_charges_vector(self, pqr_positions, pqr_charges):

        pqr_units = []
        for pos in pqr_positions:
            pqr_units.append(np.divide(pos, la.norm(pos)))
        elf_units = []
        for pos in self.positions:
            elf_units.append(np.divide(pos, la.norm(pos)))

        for i in range(len(pqr_positions)):
            if pqr_charges[i] != 0:
                min = 1000
                index = 0
                for j in range(len(elf_units)):
                    dist = la.norm(np.subtract(elf_units[j], pqr_units[i]))
                    if dist < min:
                        min = dist
                        index = j
                self.charges[index] += pqr_charges[i]
                print(pqr_charges[i], min)
