from __future__ import division
import numpy as np
import hoomd
from hoomd import md
from Loggable import Loggable


class Bonds(Loggable):

    def __init__(self, log_list=None):

        super(Bonds, self).__init__(log_list)
        #self.log_values = ['bond_fene_energy']
        self.log_values = ['bond_harmonic_energy']
        self.names = []
        self.k = []
        self.r0 = []
        self.bond_ref = None

        #self.k.append(12)
        self.k.append(120)
        #self.r0.append(1.0)
        self.r0.append(0.5)
        self.names.append('polybond')

        #self.k.append(12)
        self.k.append(120)
        self.r0.append(1.0)
        self.names.append('sidechain')

        self.k.append(120)
        self.r0.append(1.0)
        self.names.append('linker')

        self.k.append(120)
        self.r0.append(0.5)
        self.names.append('linker2')

    def set_all_bonds(self, system):
        """

        :param system: the system that needs the parameters set
        :return: reference to the harmonic bond object
        """

        #self.bond_ref = hoomd.md.bond.fene()
        self.bond_ref = hoomd.md.bond.harmonic()
        self.add_to_logger()
        snap = system.take_snapshot(all=True)
        for b in snap.bonds.types:
            name = str(b)
            #self.bond_ref.bond_coeff.set(name, k=self.k[self.names.index(name)], r0=self.r0[self.names.index(name)]
                                        # , epsilon=0.5, sigma=0.3)
            self.bond_ref.bond_coeff.set(name, k=self.k[self.names.index(name)], r0=self.r0[self.names.index(name)])
        del snap
        return self.bond_ref


