from __future__ import division
import gsd.hoomd
import gsd.fl
import numpy as np
import numpy.linalg as la
import os.path
import networkx as nx
import copy as cp
from matplotlib import pyplot as plt
plt.rcParams.update({'font.size': 22})
import hoomd.data as hbox
from mpl_toolkits.mplot3d import Axes3D
import math as m
import MDAnalysis as mda
from MDAnalysis.analysis import distances


class Analysis(object):

    def __init__(self, gsd_name, map_name):
        f = gsd.fl.open(name=gsd_name, mode='rb', application='', schema='hoomd',
                        schema_version=[1, 0])
        self.trajectory = gsd.hoomd.HOOMDTrajectory(f)
        self.universe = mda.Universe(gsd_name)
        self.mdatrajectory = self.universe.trajectory
        self.cages = []
        self.graft_sites = []
        self.grafted_chain_sequences = []
        self.grafted_chain_indices = []
        self.free_chain_sequences = []
        self.free_chain_indices = []
        self.qpi = []
        self.qmi = []
        self.read_map(map_name)
        self.frames = []
        self.gsd_name = gsd_name
        self.linker_indices = self.get_linker_indices()

    def get_linker_indices(self):

        frame0 = self.trajectory.read_frame(0)
        tipes = frame0.particles.types
        #print(tipes)
        linker_indices = []
        for i in range(len(frame0.particles.position)):
            if tipes[frame0.particles.typeid[i]] == "L":
                #print(i, tipes[frame0.particles.typeid[i]] )
                linker_indices.append(i)
        #print(len(linker_indices))
        return linker_indices


    def get_position(self, tag, frame_index):

        frame = self.trajectory.read_frame(frame_index)
        box = frame.configuration.box
        box_matrix = np.array([[box[0], 0, 0],
                      [box[3] * box[0], box[1], 0],
                      [box[4] * box[0], box[5] * box[1], box[2]]])
        pos = frame.particles.position[tag]
        #print(tag, frame.particles.image[tag])
        image = np.matmul(box_matrix, frame.particles.image[tag])

        return np.add(pos, image)

    def distance(self, tag1, tag2,  frame_index):

        pos1 = self.get_position(tag1, frame_index)
        pos2 = self.get_position(tag2, frame_index)
        return la.norm(np.subtract(pos1, pos2))


    def center_center_distances(self, frame_indeces):

        tags = []
        dists = []
        frame = self.trajectory.read_frame(0)
        for tag in range(len(frame.particles.position)):
            if self.is_center(tag, frame):
                tags.append(tag)
        for frame_index in frame_indeces:
            poses = []
            for tag in tags:
                poses.append(self.get_position(tag, frame_index))
            for ind, pos1 in enumerate(poses):
                for pos2 in poses[ind+1:]:
                    dists.append(la.norm(np.subtract(pos1, pos2)))
        return dists


    def is_center(self, tag, frame):

        tips = frame.particles.types
        if tips[frame.particles.typeid[tag]][:6] == "center":
            return True
        return False

    def graph_center_rdf(self, frame_indices, save_name="rdf.png"):

        dists = self.center_center_distances(frame_indices)
        rdf_hist, rbe = np.histogram(dists, bins=1000)
        bin_middles = [(rbe[i] + rbe[i + 1]) / 2 for i in range(len(rbe) - 1)]
        fig = plt.figure()

        ax1 = fig.add_subplot(111)

        ax1.set_title("Frames " + str(frame_indices[-1]))

        ax1.set_xlabel('R')
        ax1.set_ylabel('')
        ax1.plot(bin_middles, rdf_hist)
        plt.savefig(save_name, bbox_inches='tight', pad_inches=.2)
        plt.show()

    def percent_satisfied_linkers(self, frame, cut=1.5, pair_list=None, write_pl=False, read_pl=False):

        satisfied = []

        if read_pl:
            pair_list = self.read_pair_list(read_pl)
            #print(pair_list)

        if pair_list is not None:
            for pair in pair_list:
                dist = self.distance(pair[0], pair[1], frame)
                if dist < cut:
                    satisfied.append(pair[0])
                    satisfied.append(pair[1])
            return len(satisfied)/ len(self.linker_indices), pair_list

        pair_list = []
        for index, link in enumerate(self.linker_indices):
            count = index + 1
            if index not in satisfied:
                is_satisfied = False
                while count < len(self.linker_indices) and not is_satisfied:
                    dist = self.distance(link, self.linker_indices[count], frame)
                    if dist < cut:
                        #print(link, self.linker_indices[count], dist)
                        if link not in satisfied:
                            satisfied.append(link)
                        if self.linker_indices[count] not in satisfied:
                            satisfied.append(self.linker_indices[count])
                        pair_list.append([link, self.linker_indices[count]])
                        is_satisfied = True
                    count += 1
        if write_pl:
            self.write_pair_list(pair_list, frame)
        return len(satisfied)/ len(self.linker_indices), pair_list

    def new_pos(self, pos, ref, frame, silence=False):

        box = frame.configuration.box

        do = False
        if la.norm(np.subtract(pos, ref)) > 10:
        #    print("start")
        #    print(pos, ref)
            do = True
        x_dist = pos[0] - ref[0]
        if x_dist > box[0] / 2:
            pos[0] = pos[0] - box[0]
        elif x_dist < - box[0]/2:
            pos[0] = pos[0] + box[0]

        y_dist = pos[1] - ref[1]
        if y_dist > box[1] / 2:
            pos[1] = pos[1] - box[1]
        elif y_dist < - box[1]/2:
            pos[1] = pos[1] + box[1]

        z_dist = pos[2] - ref[2]
        if z_dist > box[2] / 2:
            pos[2] = pos[2] - box[2]
        elif z_dist < - box[2]/2:
            pos[2] = pos[2] + box[2]
        if do:
         #   print(pos, ref)
         #   print(la.norm(np.subtract(pos, ref)))
            if la.norm(np.subtract(pos,ref))>10 and not silence:
                print("aaaaaaaa")
         #   print("done")
        return pos

    def get_positions_chain(self, indices, frame):

        positions = []
        for number, index in enumerate(indices):
            position = frame.particles.position[index]
            if number != 0:
                position = self.new_pos(position, frame.particles.position[indices[number-1]], frame)
            positions.append(position)

        return positions

    def free_polymer_rgs(self, frame_index):

        rgs = []
        frame = self.trajectory.read_frame(frame_index)
        for j, chain in enumerate(self.free_chain_indices):
            positions = self.get_positions_chain(chain, frame)
            com = np.average(positions, axis=0)
            dist_arr = distances.distance_array(np.array(positions), np.array(com))
            squared = np.multiply(dist_arr, dist_arr)
            rgs.append(np.sqrt(np.sum(squared) / len(positions)))

        return rgs


    def write_pair_list(self, pair_list, frame):

        f = open(self.gsd_name + "_frame" + str(frame) + ".pl", "w")
        for pair in pair_list:
            f.write(str(pair[0]) + " " + str(pair[1]) + "\n")
        f.close()

    def read_pair_list(self, name):
        f = open(name, "r")
        data = f.readlines()
        pair_list = []
        for line in data:
            s = line.split()
            pair_list.append([int(s[0]), int(s[1])])
        return pair_list


    def read_map(self, map_name):

        f = open(map_name, 'r')

        data = f.readlines()
        graft_count = 0
        graft_seq = []
        graft_indices = []
        for line in data:
            s = line.split()
            if s[0] == "cage":
                indices = [int(ind) for ind in s[1:]]
                self.cages.append(indices)
            elif s[0] == "free_polymer":
                mons = []
                indices = []
                for ind in range(1, len(s), 2):
                    mons.append(s[ind])
                for ind in range(2, len(s), 2):
                    indices.append(int(s[ind]))
                self.free_chain_sequences.append(mons)
                self.free_chain_indices.append(indices)
            elif s[0] == "grafted_polymer":
                mons = []
                indices = []
                for ind in range(1, len(s), 2):
                    mons.append(s[ind])
                for ind in range(2, len(s), 2):
                    indices.append(int(s[ind]))
                graft_seq.append(mons)
                graft_indices.append(indices)
                graft_count += 1
                if graft_count == len(self.graft_sites[-1]):
                    self.grafted_chain_sequences.append(graft_seq)
                    self.grafted_chain_indices.append(graft_indices)
                    graft_seq = []
                    graft_indices = []
                    graft_count = 0
            elif s[0] == "graft_sites":
                indices = [int(ind) for ind in s[1:]]
                self.graft_sites.append(indices)
            elif s[0] == "qpi":
                indices = [int(ind) for ind in s[1:]]
                self.qpi.append(indices)
            elif s[0] == "qmi":
                indices = [int(ind) for ind in s[1:]]
                self.qmi.append(indices)

    def cage_valencies(self, frame_index, cut=6.5):

        time = self.mdatrajectory[frame_index].time
        frame = self.trajectory.read_frame(frame_index)
        cage_positions = [frame.particles.position[i] for i in range(len(self.cages))]
        poly_positions = []

        for chain in self.free_chain_indices:
            poly_positions.extend([frame.particles.position[i] for i in chain])

        dist_arr = distances.distance_array(np.array(cage_positions), np.array(poly_positions),
                                            box=self.universe.dimensions)

        valencies = []
        for cage in range(len(self.cages)):
            mon_count = 0
            sub_array = dist_arr[cage]
            touchers = []
            for chain_ind, chain in enumerate(self.free_chain_indices):
                for i in range(len(chain)):
                    if sub_array[mon_count] < cut:
                        if chain_ind not in touchers:
                            touchers.append(chain_ind)
                    mon_count += 1
            valencies.append(len(touchers))
        return valencies

    def cage_touchers(self, frame_index, cut=6.5):

        time = self.mdatrajectory[frame_index].time
        frame = self.trajectory.read_frame(frame_index)
        cage_positions = [frame.particles.position[i] for i in range(len(self.cages))]
        poly_positions = []

        for chain in self.free_chain_indices:
            poly_positions.extend([frame.particles.position[i] for i in chain])

        dist_arr = distances.distance_array(np.array(poly_positions), np.array(cage_positions),
                                            box=self.universe.dimensions)
        touched = 0
        for i in range(len(poly_positions)):
            sub_array = dist_arr[i]
            if cut > np.min(sub_array):
                touched += 1

        return touched, len(poly_positions)

    def graph_free_polymer_rgs(self, frame_indices, save_name="rgs.png"):

        rgs = []
        for frame_index in frame_indices:
            rgs.append(np.average(self.free_polymer_rgs(frame_index)))

        fig = plt.figure()

        ax1 = fig.add_subplot(111)

        ax1.set_title("Free Polymer Rg")

        ax1.set_xlabel('Frame')
        ax1.set_ylabel('Rg')
        ax1.plot(frame_indices, rgs)
        plt.savefig(save_name, bbox_inches='tight', pad_inches=.2)
        plt.show()

    def graph_cage_valencies(self, frame_indices, save_name="cage_valencies.png"):

        cvs = []
        for frame_index in frame_indices:
            cvs.append(np.average(self.cage_valencies(frame_index)))

        fig = plt.figure()

        ax1 = fig.add_subplot(111)

        ax1.set_title("Average Cage Valency")

        ax1.set_xlabel('Frame')
        ax1.set_ylabel('Valency')
        ax1.plot(frame_indices, cvs)
        plt.savefig(save_name, bbox_inches='tight', pad_inches=.2)
        plt.show()

    def graph_cage_touchers(self, frame_indices, save_name="cage_touchers.png"):

        pts = []
        for frame_index in frame_indices:
            touchers, total = self.cage_touchers(frame_index)
            pts.append(touchers/total)

        fig = plt.figure()

        ax1 = fig.add_subplot(111)

        ax1.set_title("Fraction Polymers Touching Cage")

        ax1.set_xlabel('Frame')
        ax1.set_ylabel('Percent')
        ax1.plot(frame_indices, pts)
        plt.savefig(save_name, bbox_inches='tight', pad_inches=.2)
        plt.show()

    def graph_linker_percent(self, frame_indices, save_name="connected_linkers.png", pair_list="pairs.pl"):

        cls = []
        for frame_index in frame_indices:
            p, pl = self.percent_satisfied_linkers(frame_index, read_pl=pair_list)
            cls.append(p)

        fig = plt.figure()

        ax1 = fig.add_subplot(111)

        ax1.set_title("Fraction Satisfied Linkers")

        ax1.set_xlabel('Frame')
        ax1.set_ylabel('Percent')
        ax1.plot(frame_indices, cls)
        plt.savefig(save_name, bbox_inches='tight', pad_inches=.2)
        plt.show()

