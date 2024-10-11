from __future__ import division
import numpy as np
from PolyAbs import PolyAbs

class SequentialPolymer(PolyAbs):

    def __init__(self, sequence, spiral=False, tight=False, peg=4, k=False, with_ion=False):

        self.spiral = spiral
        self.peg=peg
        self.k = k
        super(SequentialPolymer, self).__init__(sequence, seperation=4 - 3 * tight)
        self.sequence = sequence

    def build_chain(self, sequence, with_ion=False):

        points = self.linear_points(len(sequence), 0.5)
        if self.spiral:
            points = self.spiral_points(len(sequence), arc=0.5, separation=self.seperation)

        for i in range(len(sequence)):
            self.position.append(points[i])
            self.type.append('B')
            self.mass.append(1)
            self.charge.append(0)
            self.body.append(-1)
            self.monomer_indexes.append([i])
            if i != 0:
                self.bond_names.append('polybond')
                self.bonds.append([i-1, i])
            if self.k:
                if i !=0 and i !=1:
                    self.angle_names.append('k'+ str(self.k))
                    self.angles.append([i-2, i-1, i])
            self.length += 1
            self.num_beads += 1

        for spot, thing in enumerate(sequence):
            imp = __import__(thing, fromlist=[''])
            if thing == "PEGMEMA":
                mon = getattr(imp, thing)(n=self.peg)
            else:
                mon = getattr(imp, thing)(with_ion=False)
            mon.add_to_polymer(self, spot, up=(spot % 2 == 0))
        self.image = [[0, 0, 0] for _ in range(len(self.position))]