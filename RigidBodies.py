from __future__ import division
import numpy as np
import hoomd
from hoomd import md
from numpy import linalg as la
from Quaternion import Quaternion

class Rigid(object):

    def __init__(self):
       self.name = 'rigid'
       self.rigid = None

    def is_center(self, name):

        if len(name) < 6:
            return False
        if name[:6] != 'center':
            return False
        return True

    def set_rigid_bodies(self, system, reinit=None, o_list=None, reset=False):

        if reset:
            self.rigid.disable()

        #if reinit is not None:
        #    self.reinit_rigids(reinit)
        #    return self.rigid
        rigids = 0
        while self.is_center(str(system.particles.get(rigids).type)):
            rigids += 1
        type_list = [[] for _ in range(rigids)]
        pos_list = [[] for _ in range(rigids)]
        count = 0

        self.rigid = hoomd.md.constrain.rigid()
        for i in range(rigids, len(system.particles)):
            part = system.particles.get(i)
            if part.body > -1 and part.body < rigids:
                pos_list[part.body].append(self.de_orient(np.array(self.get_position(system,i)),
                                          np.array(self.get_position(system, part.body)),
                                           system.particles.get(part.body).orientation))
                if part.body == rigids - 1:
                    count += 1
                type_list[part.body].append(str(part.type))
        if o_list is not None:
            while len(type_list) > len(o_list):
                o_list.append([1, 0, 0, 0])
        for i in range(rigids):
            this_pos = np.array(system.particles.get(i).position)
            if len(pos_list[i]) > 0 and i==0:
                self.rigid.set_param(system.particles.get(i).type, positions=np.subtract(pos_list[i],this_pos), types=type_list[i])
                #print(system.particles.get(i).type, np.subtract(pos_list[i],this_pos), type_list[i])
            #for ind, x in enumerate(pos_list[i]):
                #print(ind, la.norm(np.subtract(x,this_pos)))
            #print(np.subtract(pos_list[i],this_pos)[8], system.particles.get(i).type)
        self.rigid.create_bodies(create=False)
        return self.rigid

    def get_rigids(self, system):
        rigids = 0
        while self.is_center(str(system.particles.get(rigids).type)):
            rigids += 1
        return rigids

    def get_spherical_radius(self, system):
        #types = system.particles.types
        for ind in range(len(system.particles)):
            type = system.particles.get(ind).type
            if type == 'qPm':
                center_ind = system.particles.get(ind).body
                return la.norm(np.subtract(system.particles.get(ind).position,
                                           system.particles.get(center_ind).position))

    def de_orient(self, position, center, orientation):

        q = Quaternion(orientation, hoomd=True).inverse()
        rel_pos = np.subtract(position, center)
        new_pos = q.orient(rel_pos)
        new_pos = np.add(new_pos, center)
        return new_pos

    def get_position(self, system, index):

        #print(system.box.Lx)
        return np.add(system.particles.get(index).position,
                      np.multiply(system.particles.get(index).image, [system.box.Lx, system.box.Ly, system.box.Lz]))


"""
    def reinit_rigids(self, file):

        import gsd.pygsd
        import gsd.hoomd
        f = gsd.pygsd.GSDFile(open(file, mode='rb'))
        frame0 = gsd.hoomd.HOOMDTrajectory(f).read_frame(0)

        types = frame0.particles.types

        for ind, pos in enumerate(frame0.particles.position):
            tipe = types[frame0.particles.typeid[ind]]
            if tipe[:6] == 'center':
                con_positions = []
                con_types = []
                center_bod = frame0.particles.body[ind]
                for ind2, pos2 in enumerate(frame0.particles.position):
                    con_tipe = types[frame0.particles.typeid[ind2]]
                    if frame0.particles.body[ind2] == center_bod and con_tipe != tipe:
                        con_positions.append(np.subtract(pos2, pos))
                        con_types.append(con_tipe)
                self.rigid.set_param(tipe, positions=con_positions, types=con_types)
        self.rigid.create_bodies(create=False)
"""

