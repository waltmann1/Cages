from __future__ import division
from Bonds import Bonds
from NonBonded import LJ
from NonBonded import Yukawa
from RigidBodies import Rigid
from Angles import Angles
import numpy as np
from numpy import linalg as la
import hoomd


class Simulation(object):

    def __init__(self, system, temperature=1, name="cages", reinit=None, o_list=None, energy=None, total_charge=None,
                 debye=1, poly_cage=None, poly_poly=None):

        if energy is None:
            raise ValueError("Must Set energy of linkers")
        self.system = system
        self.nlist = hoomd.md.nlist.cell(check_period=1)
        self.nlist.reset_exclusions(exclusions=['bond', 'angle', 'dihedral', 'constraint', 'body'])
        #self.log_list = ['potential_energy', 'temperature', 'kinetic_energy']
        self.log_list = ['potential_energy', 'ndof', 'kinetic_energy', 'pressure', 'volume']
        self.log_list.append('temperature')
        self.log_period = 100
        self.dump_period = 1000
        self.temperature = temperature
        self.name = name
        self.energy = energy
        self.poly_cage = poly_cage
        self.poly_poly = poly_poly

        if o_list is not None:
            for i in range(len(o_list)):
                self.system.particles[i].orientation = o_list[i]

        self.dt = .001

        self.rigid = Rigid()
        self.rigid.set_rigid_bodies(system, reinit=reinit, o_list=o_list)

        self.bonds = Bonds(self.log_list)
        self.bonds.set_all_bonds(system)

        self.angles = Angles(self.log_list)
        self.angles.set_all_harmonic_angles(system)

        self.lj = LJ(self.log_list, energy=energy)
        self.lj.set_lj(self.nlist, system)


        #self.charger = ThreePM(self.log_list)
        #self.charger.set_charges(self.nlist, self.system, lamda=lamda)

        if total_charge is not None:
            self.charger = Yukawa(self.log_list, total_charge=total_charge, debye=debye)
            self.charger.set_yukawa(self.nlist, self.system)

        self.all = hoomd.group.all()

        self.to_integrate = hoomd.group.union(name='dof', a=hoomd.group.rigid_center(), b=hoomd.group.nonrigid())
        #self.uncharged = hoomd.group.difference(name='uncharged', a=self.to_integrate, b=hoomd.group.charged())
        self.poly_group = hoomd.group.type("P", name="polymers")

        hoomd.md.integrate.mode_standard(dt=self.dt)
        self.npt = hoomd.md.integrate.npt(group=self.to_integrate, kT=1, tau=1.0, tauP=1, P=2e-3)
        self.npt.disable()
        self.nve = hoomd.md.integrate.nve(group=self.to_integrate, limit=.001)
        self.nve.disable()
        self.langevin = hoomd.md.integrate.langevin(group=self.to_integrate, kT=self.temperature, seed=42)

        log_name = self.name + ".log"
        self.logger = hoomd.analyze.log(filename=log_name, quantities=self.log_list, period=self.log_period,
                          overwrite=True)

        dump_name = self.name + ".gsd"
        self.dumper = hoomd.dump.gsd(filename=dump_name, period=self.dump_period, group=self.all, overwrite=True)

    def run(self, time):

        #print(self.system.constraints)
        hoomd.run(time)

    def run_polymers(self, time):

        self.langevin.disable()
        temp = hoomd.md.integrate.langevin(group=self.poly_group, kT=self.temperature, seed=42)
        hoomd.run(time)
        temp.disable()
        self.langevin = hoomd.md.integrate.langevin(group=self.to_integrate, kT=self.temperature, seed=42)

    def zero_momentum(self):

        hoomd.md.update.zero_momentum()

    def run_nanoseconds(self, time):

        real_time = int(time * 1e-9 / (self.time_unit * self.dt))
        self.run(real_time)

    def nve_relaxation(self, time, limit=.001):

         self.langevin.disable()
         self.nve.enable()
         self.nve.set_params(limit=limit)
         hoomd.run(time)
         self.nve.disable()
         self.langevin.enable()

    def enable_npt(self):

        self.langevin.disable()
        self.npt.enable()

    def npt_temp_interp(self, temp1, temp2, time):

        t1 = temp1
        t2 = temp2
        self.npt.set_params(kT=hoomd.variant.linear_interp(points=[(0, t1), (time, t2)]))
        hoomd.run(time)
        self.npt.set_params(kT=self.temperature)

    def disable_npt(self):

        self.npt.disable()
        self.langevin.enable()

    def set_dt(self, dt):
        hoomd.md.integrate.mode_standard(dt=dt)

    def run_fire(self, time):

        self.langevin.disable()
        self.nve.enable()
        fire = hoomd.md.integrate.mode_minimize_fire(dt=0.1, group=self.to_integrate, ftol=1e-2, Etol=1e-7)
        hoomd.run(time)
        del fire
        self.langevin.enable()
        self.nve.disable()
        hoomd.md.integrate.mode_standard()

    def temp_interp(self, temp1, temp2, time):

        t1 = temp1
        t2 = temp2
        self.langevin.set_params(kT=hoomd.variant.linear_interp(points=[(0, t1), (time, t2)]))
        hoomd.run(time)
        self.langevin.set_params(kT=self.temperature)

    def set_temperature(self, kt):
        self.temperature = kt
        self.langevin.set_params(kT=self.temperature)

    def basic_temp_equil_no_log(self):

        self.logger.disable()
        self.dumper.disable()
        self.set_temperature(0)
        self.run(10000)
        self.temp_interp(0, 1, 100000)
        self.set_temperature(1)
        self.run(10000)
        self.logger.enable()
        self.dumper.enable()

    def set_log_period(self, period):

        self.logger.disable()
        self.log_period = period
        log_name = self.name + ".log"
        self.logger = hoomd.analyze.log(filename=log_name, quantities=self.log_list, period=self.log_period,
                                        overwrite=True)

    def set_dump_period(self, period):

        self.dumper.disable()
        self.dump_period = period
        dump_name = self.name + ".gsd"
        self.dumper = hoomd.dump.gsd(filename=dump_name, period=self.dump_period, group=self.all, overwrite=True)


    def total_kinetic_energy(self):

        ke = 0
        for part in self.system.particles:
            kin = .5 * part.mass * np.linalg.norm(part.velocity) ** 2
            print(part.type, kin)
            ke += kin


        return ke

    def ndof(self):

        return self.total_kinetic_energy() * 2 / self.temperature

    def update_log_list(self):

        self.logger.disable()
        log_name = self.name + ".log"
        self.logger = self.logger = hoomd.analyze.log(filename=log_name, quantities=self.log_list, period=self.log_period,
                          overwrite=True)

    def turn_off_charge(self):
        self.charger.turn_off_charge()
        self.set_dt(.002)
        self.update_log_list()

    def turn_on_charge(self):
        self.charger.turn_on_charge()
        self.update_log_list()

    def set_total_charge(self, total_charge):

        self.turn_off_charge()
        self.charger.total_charge = total_charge
        self.charger.set_yukawa(self.nlist, self.system)

    def set_effective_charge(self, effective_charge):

        self.turn_off_charge()
        self.charger.effective_charge = effective_charge
        self.charger.set_yukawa(self.nlist, self.system)

    def anneal(self, total_time, max_temp=2):
        time = total_time / 2

        self.temp_interp(1, max_temp, time)
        self.temp_interp(max_temp, 1, time)

    def integrate_dyes_only(self, num_dyes):

        dye_group = hoomd.group.tag_list(name="dyes", tags=[i for i in range(num_dyes)])
        self.langevin.disable()
        self.langevin = hoomd.md.integrate.langevin(group=dye_group, kT=self.temperature, seed=42)

    def box_interp(self, old_size, new_size, time):

        up = hoomd.update.box_resize(L=hoomd.variant.linear_interp(points=[(0, old_size), (time, new_size)]))
        hoomd.run(time)
        up.disable()

    def set_energy(self, energy):

        self.energy = energy
        self.reset_lj()

    def set_poly_poly(self, pp):

        self.poly_poly = pp

    def set_poly_cage(self, pc):

        self.poly_cage = pc

    def set_debye_length(self, debye):

        self.charger.turn_off_charge()
        self.charger.kappa = (1 / debye)
        self.charger.set_yukawa(self.nlist, self.system)
        self.charger.turn_on_charge()

    def reset_lj(self):
        self.lj.disable()
        self.lj = LJ(self.log_list, energy=self.energy)
        self.lj.poly_poly = self.poly_poly
        self.lj.poly_cage = self.poly_cage
        self.lj.set_lj(self.nlist, self.system)


class InitGSD(Simulation):

    def __init__(self, name, frame, energy=1, lamda=1):

        hoomd.context.initialize("--mode=gpu")
        system = hoomd.init.read_gsd(name, frame=frame)
        i = 0

        while not name[i].isalpha():
            i = i + 1

        name_no_loc = name[i:]
        #quit()
        super(InitGSD, self).__init__(system, name=name_no_loc[:-4] + '_frame' + str(frame), energy=energy, lamda=lamda)


class MinImageGSD(Simulation):

    def __init__(self, name, frame, map_name,  energy=1, lamda=1):

        hoomd.context.initialize("--mode=gpu")

        system = hoomd.init.read_gsd(name, frame=frame)
        self.system = system.take_snapshot()
        self.min_image(map_name)
        self.system.box = hoomd.data.boxdim(L=200)

        print(self.system.box)
        #quit()

        system.restore_snapshot(self.system)

        self.system = system
        i =0
        while not name[i].isalpha():
            i = i + 1

        name_no_loc = name[i:]

        super(MinImageGSD, self).__init__(self.system, name=name_no_loc[:-4] + '_frame' + str(frame), energy=energy, lamda=lamda)

    def min_image(self, map_name):

        #hoomd.update.box_resize(L=200)
        self.box = [self.system.box.Lx, self.system.box.Ly, self.system.box.Lz]
        #self.system.particles.position[0] = [0,0,0]
        self.read_map(map_name)
        for chain in self.all_chain_indices:
            n = len(chain)
            middle = int(np.floor(n/2))
            for i in range(middle + 1, n):
                ref = self.system.particles.position[chain[i-1][0]]
                pos = self.system.particles.position[chain[i][0]]
                self.system.particles.position[chain[i][0]] = self.new_pos(pos, ref)
                for j in range(len(chain[i]) - 1):
                    ref = self.system.particles.position[chain[i][j]]
                    pos = self.system.particles.position[chain[i][j+1]]
                    #save_pos = [pos[0]+1-1, pos[1], pos[2]]
                    self.system.particles.position[chain[i][j+1]] = self.new_pos(pos, ref)
                    #self.system.particles.get(chain[i][j+1]).position = [100,100,100]
            for i in range(middle, 0, -1):
                ref = self.system.particles.position[chain[i][0]]
                pos = self.system.particles.position[chain[i-1][0]]
                self.system.particles.position[chain[i-1][0]] = self.new_pos(pos, ref)
                for j in range(len(chain[i-1]) - 1):
                    ref = self.system.particles.position[chain[i-1][j]]
                    pos = self.system.particles.position[chain[i-1][j+1]]
                    self.system.particles.position[chain[i-1][j + 1]] = self.new_pos(pos, ref)

        for dye in self.dyes:
            for i in range(0,len(dye)):
                ref = self.system.particles.position[dye[0]]
                pos = self.system.particles.position[dye[i]]
                if self.system.particles.types[self.system.particles.typeid[dye[i]]][-1] != 'i':
                    self.system.particles.position[dye[i]] = self.new_pos(pos, ref)


    def new_pos(self, pos, ref):

        pos = list(pos)
        ref = list(ref)
        save = [pos[0] + 1 -1, pos[1], pos[2]]
        do = False
        if la.norm(np.subtract(pos, ref)) > 10:
            do = True
        x_dist = pos[0] - ref[0]
        if x_dist > self.box[0] / 2:
            pos[0] = pos[0] - self.box[0]
        elif x_dist < - self.box[0]/2:
            pos[0] = pos[0] + self.box[0]

        y_dist = pos[1] - ref[1]
        if y_dist > self.box[1] / 2:
            pos[1] = pos[1] - self.box[1]
        elif y_dist < - self.box[1]/2:
            pos[1] = pos[1] + self.box[1]

        z_dist = pos[2] - ref[2]
        if z_dist > self.box[2] / 2:
            pos[2] = pos[2] - self.box[2]
        elif z_dist < - self.box[2]/2:
            pos[2] = pos[2] + self.box[2]
        if do:
            if la.norm(np.subtract(pos,ref))>10:
                print("aaaaaaaa")
        return pos

    def read_map(self, map_name):

        f = open(map_name, 'r')

        dyes = []
        rando_chain_sequences = []
        rando_chain_indices = []
        charged_chain_indices = []
        charged_chain_sequences = []

        data = f.readlines()

        for line in data:
            s = line.split()
            if s[0] == "dye":
                dyes.append([int(float(s[i])) for i in range(1, len(s))])
            elif s[0] == "rando_polymer":
                i = 2
                mon = []
                chain = []
                sequence = [s[1]]
                while i < len(s):
                    if not any(char.isalpha() for char in s[i]):
                        mon.append(int(float(s[i])))
                    else:
                        chain.append(mon)
                        mon = []
                        sequence.append(s[i])
                    i += 1
                chain.append(mon)
                rando_chain_indices.append(chain)
                rando_chain_sequences.append(sequence)
            elif s[0] == "charged_polymer":
                i = 2
                sequence = [s[1]]
                mon = []
                chain = []
                while i < len(s):
                    if not s[i].isalpha():
                        mon.append(int(float(s[i])))
                    else:
                        chain.append(mon)
                        mon = []
                        sequence.append(s[i])
                    i += 1
                chain.append(mon)
                charged_chain_indices.append(chain)
                charged_chain_sequences.append(sequence)
            elif s[0] == "qpi":
                qpi = [int(float(s[i])) for i in range(1, len(s))]
            elif s[0] == "qmi":
                qmi = [int(float(s[i])) for i in range(1, len(s))]

        self.dyes = dyes
        self.rando_chain_indices = rando_chain_indices
        self.charged_chain_indices = charged_chain_indices
        self.all_chain_indices = rando_chain_indices + charged_chain_indices
        self.qmi = qmi
        self.qpi = qpi



