from SphereCage import SphereCage
from FerritinCage import FerritinCage
from SequentialPolymer import SequentialPolymer
from CageLattice import CageLattice
import numpy as np
from Solution import Solution
import copy as cp
from Simulation import Simulation


c = FerritinCage(5.4,6, qxyz="ferr5_46_pH6_5.qxyz")
#c.dump_qxyz("ferr5_46_pH6_5")

#c = SphereCage(6)


#c.add_six_fold_graft_sites()
c.add_neighbor_linkers(points="fcc")


sequence = ['MinusMonomer' for _ in range(0)] + ['PlusMonomer' for _ in range(50)] +\
           ['NeutralMonomer' for _ in range(0)]

p = SequentialPolymer(sequence)

polys = [cp.deepcopy(p) for _ in range(200)]
#polys = []

abc = [18, 18, 18]
units = [2, 2, 2]
bcc_points = [[0, 0, 0], [.5, .5, .5]]
fcc_points = [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]]
sc_points = [[0, 0, 0]]
cubic_vectors = np.eye(3)
hex_vectors = [[1, 0, 0], [-.5, np.sqrt(3)/2, 0], [0, 0, 1]]
lattice_vectors = np.multiply(cubic_vectors, abc)
points = fcc_points
points = np.add(points, [0.5, 0.5, 0.5])
lattice_points = np.multiply(points, abc)



s = CageLattice(c, polys, lattice_vectors, lattice_points, units)

#s = Solution([c],polys, box_length=19, cell_matrix=np.multiply(19,cubic_vectors))

#s.dump_xyz("cage")
#s.dump_gsd("cage")

sim = Simulation(s.create_system(), energy=30)
#sim.run(1)
sim.set_dump_period(1e3)
sim.nve_relaxation(5e3)