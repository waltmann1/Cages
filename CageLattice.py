from __future__ import division
import numpy as np
from Solution import Solution
import copy as cp
import hoomd
import numpy.linalg as la


class CageLattice(Solution):

    def __init__(self, cage, chains, lattice_vectors, lattice_points, units, calcium=0):

        """

        :param cages: normal for a solution
        :param chains: normal for a solution
        :param lattice_vectors:  unit cell parameters a,b,c
        :param lattice_points: cage positions, must include [0,0,0]
        :param units: vector of length 3, how many in each direction
        """

        total_points = []
        for x in range(units[0]):
            for y in range(units[1]):
                for z in range(units[2]):
                    vectors = np.transpose(np.multiply([x, y, z], np.transpose(lattice_vectors)))
                    new_points = lattice_points
                    for vector in vectors:
                        new_points = np.add(new_points,vector)
                    total_points.extend(new_points)

        box_vectors = [np.multiply(lattice_vectors[i], units[i]) for i in range(3)]
        cell_matrix = np.transpose(box_vectors)
        point_bottom = np.divide([np.sum(cell_matrix[i]) for i in range(3)], -2)
        total_points = np.add(total_points, point_bottom)
        cages = []
        for point in total_points:
            new_cage = cp.deepcopy(cage)
            new_cage.shift(point)
            cages.append(new_cage)

        super(CageLattice, self).__init__(cages, chains, box_length=0, cell_matrix=cell_matrix, calcium=calcium)

        self.add_chains(graft=True)

        for cage in self.cages:
            cage.enforce_generic_bc(cell_matrix)
        for chain in self.chains:
            chain.enforce_generic_bc(cell_matrix)

        #for cage in self.cages:
        #    cage.enforce_cubic_bc(box_lengths[0])
        #for chain in self.chains:
        #    chain.enforce_cubic_bc(box_lengths[0])




