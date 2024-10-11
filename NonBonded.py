from __future__ import division
import numpy as np
import hoomd
from hoomd import md
from numpy import linalg as la
from Loggable import Loggable


class LJ(Loggable):
    def __init__(self, log_list=None, energy=30, poly_poly=None, poly_cage=None):

        super(LJ, self).__init__(log_list)

        self.log_values = ['pair_lj_energy']

        self.epsilon = [1, 1, 1, 1, 1]
        self.sigma = [0.5, 1.0, 1.0, 1.0, 0.5]
        #self.sigma = [0.5, 0.5, 0.5, 0.5, 0.5, 0.3, 0.3]
        self.names = ['P', 'M', 'N', 'L', 'Ca']

        self.lj_pair = None

        self.energy = energy

        self.poly_cage = poly_cage
        self.poly_poly = poly_poly

    def set_lj(self, neighbor_list, system):

        if self.poly_cage == 0:
            self.poly_cage = None
        if self.poly_poly == 0:
            self.poly_poly = None

        cut = 3
        self.lj_pair = hoomd.md.pair.lj(r_cut=cut, nlist=neighbor_list)
        self.add_to_logger()
        for t1 in system.particles.types:
            for t2 in system.particles.types:
                t1 = str(t1)
                t2 = str(t2)
                #print(t1, t2)
                self.lj_pair.pair_coeff.set(str(t1), str(t2), epsilon=1, sigma=1, r_cut=1)

                if t1[0] == 'L' and t2[0] == 'L':
                    if len(t1) > 1 and len(t2) > 1:
                        self.lj_pair.pair_coeff.set(str(t1), str(t2), epsilon=self.energy, sigma=1, r_cut=3)
                    else:
                        print("a")
                        sig = (self.sigma[self.names.index(t1)] + self.sigma[self.names.index(t2)]) / 2
                        cut = 3 * sig
                        eps = np.sqrt(self.epsilon[self.names.index(t1)] * self.epsilon[self.names.index(t2)])
                        self.lj_pair.pair_coeff.set(str(t1), str(t2), epsilon=self.energy * eps, sigma=sig, r_cut=cut)
                        print("cut", cut)
                elif t1 == "N" and t2 == "N":
                    eps = np.sqrt(self.epsilon[self.names.index(t1)] * self.epsilon[self.names.index(t2)])
                    sig = (self.sigma[self.names.index(t1)] + self.sigma[self.names.index(t2)]) / 2
                    sig += 1
                    WCA_cut = 2 ** (1.0 / 6.0) * sig
                    self.lj_pair.pair_coeff.set(str(t1), str(t2), epsilon=eps, sigma=sig, r_cut=WCA_cut)
                elif t1 in self.names and t2 in self.names:
                    print("b")
                    eps = np.sqrt(self.epsilon[self.names.index(t1)] * self.epsilon[self.names.index(t2)])
                    sig = (self.sigma[self.names.index(t1)] + self.sigma[self.names.index(t2)]) / 2
                    WCA_cut = 2 ** (1.0 / 6.0) * sig
                    if self.poly_cage is not None and ((t1 == "P" and t2 == "N") or (t1 == "N" and t2 == "P")):
                        eps = self.poly_cage
                        WCA_cut = 3 * sig
                    if self.poly_poly and t1 == "P" and t2 == "P":
                        eps = self.poly_poly
                        WCA_cut = 3 * sig
                    self.lj_pair.pair_coeff.set(str(t1), str(t2), epsilon=eps, sigma=sig, r_cut=WCA_cut)
                    print(WCA_cut)
                else:
                    print("c")
                    sig = 1
                    self.lj_pair.pair_coeff.set(str(t1), str(t2), epsilon=0, sigma=sig, r_cut=1)


    def disable(self):
        self.lj_pair.disable()


    def is_center(self, string):

        if len(string) >= 6 and string[:6] == 'center':
            return True
        return False


class Yukawa(Loggable):

    def __init__(self, log_list=None, debye=1, total_charge=None, effective_charge=1):
        super(Yukawa, self).__init__(log_list)
        self.log_values = ['pair_yukawa_energy']
        self.lb = .7
        self.sigma = [0.5, 1.0, 1.0, 1.0, 0.5]
        self.names = ['P', 'M', 'N', 'L', 'Ca']
        self.charge = [1, 0, 0, 0, -2]
        self.kappa = 1/debye
        self.yukawa = None
        self.total_charge = total_charge
        self.effective_charge = effective_charge

    def set_yukawa(self, neighbor_list, system):

        yuk = hoomd.md.pair.yukawa(r_cut=3 / self.kappa, nlist=neighbor_list)
        print("self.kappa", self.kappa)
        self.add_to_logger()
        self.nlist = neighbor_list
        self.system = system

        if self.total_charge is not None:
            count = 0
            for t in system.particles:
                t = str(t.type)
                if t == 'N':
                    count += 1
            if count > 0:
                self.charge[self.names.index('N')] = self.total_charge / count
            else:
                self.charge[self.names.index('N')] = 0
        for t1 in system.particles.types:
            for t2 in system.particles.types:
                t1 = str(t1)
                t2 = str(t2)
                if t1 in self.names and t2 in self.names:
                    sigma = .5 * (self.sigma[self.names.index(t1)] + self.sigma[self.names.index(t2)])
                    q1 = self.charge[self.names.index(t1)] * self.effective_charge
                    q2 = self.charge[self.names.index(t2)] * self.effective_charge
                    #eps = q1 * q2 * self.lb * np.exp(self.kappa * sigma) / (1/self.kappa + sigma)
                    eps = q1 * q2 * self.lb
                    #eps = eps * 100
                    yuk.pair_coeff.set(t1, t2, epsilon=eps, kappa=self.kappa)
                else:
                    yuk.pair_coeff.set(t1, t2, epsilon=0, kappa=self.kappa)
        self.yukawa = yuk

    def turn_off_charge(self):

        if self.yukawa is not None:
            self.yukawa.disable()
            self.remove_from_logger()

    def turn_on_charge(self):

        if self.yukawa is not None:
            self.yukawa.enable()
            self.add_to_logger()


class ThreePM(Loggable):
    def __init__(self, log_list=None, eps_relative=None, lb=None):

        super(ThreePM, self).__init__(log_list)

        if eps_relative is None and lb is None:
            eps_relative = 80

        if eps_relative is None and lb is not None:
            eps_relative = 56/lb

        if lb is None and eps_relative is not None:
            lb = 56/ eps_relative

        if lb is not None and eps_relative is not None and eps_relative * lb != 56:
            raise ValueError("lb and eps_relative are inconsistent. Must multiply to 56")


        self.log_values = ['pppm_energy']
        self.charge = [0, 0, 0, -1, 1, -1, -1, 1, 0]
        self.names = ['Bb', 'B', 'L', 'QM', 'QP', 'P', 'L', 'N', 'center']
        self.charge_adjusted = False
        # self.charge_unit = 1.6 * 10^-19 / sqrt( 4 pi eps_0 eps_r  4.11 * 10^-21 J/kt 10^-9 m )
        self.charge_unit = .837 * np.sqrt(80) / np.sqrt(eps_relative)

        self.object = None

    def set_charges(self, neighbor_list, system, lamda=1):

        found_charge = False
        for part in system.particles:
            if self.charge[self.names.index(part.type)] != 0:
                found_charge = True
                if not self.charge_adjusted:
                    part.charge = part.charge * self.charge_unit * lamda

        if found_charge:
            self.object = hoomd.md.charge.pppm(group=hoomd.group.charged(), nlist=neighbor_list)
            #self.object = hoomd.md.charge.pppm(group=hoomd.group.all(), nlist=neighbor_list)
            self.add_to_logger()
            #self.object.set_params(Nx=32, Ny=32, Nz=32, order=5, rcut=5 * 2 ** (1/6))
            self.object.set_params(Nx=64, Ny=64, Nz=64, order=6, rcut=3 * 2 ** (1 / 6))
            self.charge_adjusted = True

        if self.charge_adjusted:
            self.charge_unit = 1

    def turn_off_charge(self):

        if self.object is not None:
            self.object.disable()
            self.remove_from_logger()

    def turn_on_charge(self, lamda=1):

        if self.object is not None:
            self.object.enable()
            self.add_to_logger()


