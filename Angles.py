from __future__ import division
import numpy as np
import hoomd
from hoomd import md
from Loggable import Loggable


class Angles(Loggable):

    def __init__(self, log_list=None):

        super(Angles, self).__init__(log_list)
        self.log_values = ['angle_harmonic_energy']
        self.names = []
        self.k = []
        self.theta = []
        self.angle_ref = None

        #self.names.append('nothing')
        #self.theta.append(np.deg2rad(180))
        #self.k.append(0)

        self.names.append('stiff')
        self.theta.append(np.deg2rad(180))
        self.k.append(10000)

        self.names.append('stiff_3')
        self.theta.append(np.deg2rad(120))
        self.k.append(10000)

        self.names.append('k1')
        self.theta.append(np.deg2rad(180))
        self.k.append(1)

        self.names.append('k10')
        self.theta.append(np.deg2rad(180))
        self.k.append(10)

        self.names.append('k100')
        self.theta.append(np.deg2rad(180))
        self.k.append(100)

        self.names.append('k5')
        self.theta.append(np.deg2rad(180))
        self.k.append(5)

        self.names.append('linker')
        self.theta.append(np.deg2rad(180))
        self.k.append(1)


    def set_all_harmonic_angles(self, system, reset=False, poly=0):

        if reset:
            self.angle_ref.disable()
        if len(system.angles) == 0:
            return

        self.angle_ref = hoomd.md.angle.harmonic()
        self.add_to_logger()
        snap = system.take_snapshot(all=True)
        for a in snap.angles.types:
            name = str(a)
            self.angle_ref.angle_coeff.set(name, k=self.k[self.names.index(name)],
                                           t0=self.theta[self.names.index(name)])
        return self.angle_ref



