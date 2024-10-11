from __future__ import division
import numpy as np
from numpy import linalg as la
import hoomd
import copy as cp
from SequentialPolymer import SequentialPolymer

class Solution(object):

    def __init__(self, cages, chains, box_length, cell_matrix=None, calcium=0):

        if cell_matrix is None:
            raise ValueError("Solution requires a cell matrix")
        self.cages = [cp.deepcopy(cage) for cage in cages]
        self.chains = [cp.deepcopy(chain) for chain in chains]
        self.add_calcium(calcium)
        self.dump_context = None
        self.rigid_count = 0
        for _ in range(len(self.cages)):
            self.rigid_count +=1
        self.num_particles = np.sum([len(cage.positions) for cage in cages]) + np.sum([len(chain.mass) for chain in self.chains])
        self.cages = self.reindex()
        self.name = "assembly"
        self.box_length = int(box_length)
        self.cell_matrix = cell_matrix
        self.o_list = None
        frac_positions = []
        mon_length = 0.5
        self.mon_length = mon_length
        poly_on_a_lattice = False
        if poly_on_a_lattice:
            num_points = [int(la.norm(self.cell_matrix[i])/mon_length) for i in range(3)]
            for i in range(num_points[0]):
                for j in range(num_points[1]):
                    for k in range(num_points[2]):
                        frac_positions.append([i/num_points[0] - .5, j/num_points[1] - .5, k/num_points[2] - .5])
            print(len(frac_positions), frac_positions)
            for cage in self.cages:
                print(cage.center)
            #quit()
            lattice_positions = list(np.matmul(frac_positions, self.cell_matrix))
            for ind, pos in enumerate(lattice_positions):
                lattice_positions[ind] = list(pos)
            print(len(lattice_positions), lattice_positions[-1], type(lattice_positions) )
            self.poly_points = []
            for pos in lattice_positions:
                if not self.clashes(pos, 6.5):
                    print(len(self.poly_points))
                    self.poly_points.append(pos)
            del frac_positions
            del lattice_positions
            print(len(self.poly_points), type(self.poly_points))

    def reindex(self):

        tag = 0
        new_list = []
        added = []
        for index, cage in enumerate(self.cages):
            cage.index = tag
            tag += 1
            new_list.append(cage)
            added.append(index)
        for index, cage in enumerate(self.cages):
            if index not in added:
                new_list.append(cage)
        return new_list

    def create_system(self):
        """

        :return: system object
        """
        if self.dump_context is None:
            self.dump_context = hoomd.context.initialize("")
        b_types = self.parse_bond_types()
        p_types = self.parse_particle_names() + ["center"]
        a_types = self.parse_angle_types()
        l = self.box_length

        box = hoomd.data.boxdim(L=l)

        if self.cell_matrix is not None:
            v = [[] for _ in range(3)]
            v[0] = self.cell_matrix[:, 0]
            v[1] = self.cell_matrix[:, 1]
            v[2] = self.cell_matrix[:, 2]
            Lx = np.sqrt(np.dot(v[0], v[0]))
            a2x = np.dot(v[0], v[1]) / Lx
            Ly = np.sqrt(np.dot(v[1], v[1]) - a2x * a2x)
            xy = a2x / Ly
            v0xv1 = np.cross(v[0], v[1])
            v0xv1mag = np.sqrt(np.dot(v0xv1, v0xv1))
            Lz = np.dot(v[2], v0xv1) / v0xv1mag
            a3x = np.dot(v[0], v[2]) / Lx
            xz = a3x / Lz
            yz = (np.dot(v[1], v[2]) - a2x * a3x) / (Ly * Lz)
            box = hoomd.data.boxdim(Lx=Lx, Ly=Ly, Lz=Lz, xy=xy, xz=xz, yz=yz)
            #box = hoomd.data.boxdim(L=19)

        snap = hoomd.data.make_snapshot(int(self.num_particles + self.rigid_count), particle_types=p_types,
                                        bond_types=b_types, angle_types=a_types,
                                        box=box)
        #snap.particles.position[0] = [-9.5, -16.454483, -19]
        #snap.particles.position[1] = [9., 16., 18.]
        #return hoomd.init.read_snapshot(snap)
        snap.bonds.resize(0)

        for x in range(self.rigid_count):
            snap.particles.position[x] = self.cages[x].center
            snap.particles.mass[x] = np.sum([self.cages[x].rigid_mass()])
            snap.particles.typeid[x] = p_types.index("center")
            snap.particles.body[x] = x
            snap.particles.moment_inertia[x] = self.cages[x].moment
            snap.particles.orientation[x] = self.cages[x].orientation

        tag = self.rigid_count
        for cage in self.cages:
            ci = cage.index
            for x in range(len(cage.bonds)):
                bond_number = snap.bonds.N + 1
                snap.bonds.resize(bond_number)
                snap.bonds.group[bond_number - 1] = np.add(cage.bonds[x], tag)
                snap.bonds.typeid[bond_number - 1] = b_types.index(cage.bond_names[x])

            for x in range(len(cage.angles)):
                angle_number = snap.angles.N + 1
                snap.angles.resize(angle_number)
                snap.angles.group[angle_number - 1] = np.add(cage.angles[x], tag)
                snap.angles.typeid[angle_number - 1] = a_types.index(cage.angle_names[x])
            for x in range(len(cage.positions)):
                snap.particles.position[x + tag] = cage.positions[x]
                snap.particles.mass[x + tag] = cage.masses[x]
                snap.particles.typeid[x + tag] = p_types.index(cage.types[x])
                if x < cage.num_rigid:
                    snap.particles.body[x + tag] = ci
                else:
                    snap.particles.body[x + tag] = -1
                snap.particles.charge[x + tag] = cage.charges[x]
            tag += len(cage.positions)

        for chain in self.chains:
            for x in range(len(chain.bonds)):
                bond_number = snap.bonds.N + 1
                snap.bonds.resize(bond_number)
                snap.bonds.group[bond_number - 1] = np.add(chain.bonds[x], tag)
                snap.bonds.typeid[bond_number - 1] = b_types.index(chain.bond_names[x])

            for x in range(len(chain.angles)):
                angle_number = snap.angles.N + 1
                snap.angles.resize(angle_number)
                snap.angles.group[angle_number - 1] = np.add(chain.angles[x], tag)
                snap.angles.typeid[angle_number - 1] = a_types.index(chain.angle_names[x])

            for x in range(chain.num_beads):
                snap.particles.position[x + tag] = chain.position[x]
                snap.particles.mass[x + tag] = chain.mass[x]
                snap.particles.typeid[x + tag] = p_types.index(chain.type[x])
                snap.particles.body[x + tag] = chain.index
                snap.particles.charge[x + tag] = chain.charge[x]
                snap.particles.image[x + tag] = chain.image[x]
            tag += chain.num_beads

        print("parameters", Lx, Ly, Lz, xy, xz, yz)
        sys = hoomd.init.read_snapshot(snap)
        self.dump_map()

        return sys

    def parse_bond_types(self):

        a_types = []
        for chain in self.chains:
            for c in chain.bond_names:
                if c not in a_types:
                    a_types.append(c)
        for cage in self.cages:
            for c in cage.bond_names:
                if c not in a_types:
                    a_types.append(c)
        return a_types

    def parse_angle_types(self):

        a_types = []
        for chain in self.chains:
            for c in chain.angle_names:
                if c not in a_types:
                    a_types.append(c)
        for cage in self.cages:
            for c in cage.angle_names:
                if c not in a_types:
                    a_types.append(c)

        return a_types

    def parse_particle_names(self):
        a_types = []
        for chain in self.chains:
            for c in chain.type:
                if c not in a_types:
                    a_types.append(c)
        for cage in self.cages:
            for d in cage.types:
                if d not in a_types:
                    a_types.append(d)

        return a_types

    def geometric_center(self):

        pos = [np.mean(c.position, axis=0) for c in self.chains]
        pos_cages = [cage.center for cage in self.cages]
        pos = pos + pos_cages
        weights = [len(c.position) for c in self.chains]
        weights_mers = [len(dye.position) for dye in self.dyes]
        weights = weights + weights_mers
        total = np.sum(weights)
        weights = [float(weight)/float(total) for weight in weights]
        center = np.array([0, 0, 0])
        for ind, p in enumerate(pos):
            #print(weights[ind])
            center = np.add(center, np.multiply(p, weights[ind]))
        return center

    def max_radius(self):

        t = [-1 * c for c in self.geometric_center()]
        self.shift(t)
        rs = [la.norm(pos) for chain in self.chains for pos in chain.position]
        self.shift([-1 * g for g in t])
        return np.max(rs)

    def shift(self, vector):

        for cage in self.cages:
            cage.shift(vector)

        for chain in self.chains:
            chain.shift(vector)

    def center_of_mass(self, center_index):
        for cage in self.cages:
            if cage.index == center_index:
                return cage.rigid_center_of_mass()

    def total_rigid_mass(self, center_index):
        for cage in self.cages:
            if cage.index == center_index:
                return cage.total_rigid_mass()
        for chain in self.chains:
            if chain.index == center_index:
                return np.sum(chain.mass)

    def moment_inertia(self, center_index):
        for cage in self.cage:
            if cage.index == center_index:
                return cage.calculate_interia_tensor()

    def dump_gsd(self, dump_file=None):
        """

        :param dump_file: name of the file to dump the xyz to
        :return: nothing
        """

        filename = dump_file
        if dump_file is None:
            filename = self.name + '.gsd'
        elif dump_file[-4:] != '.gsd':
            filename = dump_file + '.gsd'

        sys = self.create_system()
        #res_map = self.create_res_map(sys)

        hoomd.dump.gsd(filename=filename, period=None, group=hoomd.group.all(), overwrite=True, dynamic=["momentum"])
        return filename

    def orient_quaternion(self, q):
        temp = self.geometric_center()
        self.shift(np.multiply(-1, temp))
        for chain in self.chains:
            for x in range(chain.num_particles):
                chain.position[x] = q.orient(chain.position[x])
        self.shift(temp)

    def center_at_origin(self):
        temp = self.geometric_center()
        self.shift(np.multiply(-1, temp))

    def dump_map(self, dump_file=None):

        qpi_list = []
        qmi_list = []
        filename = dump_file
        if dump_file is None:
            filename = self.name + '.map'
        elif dump_file[-4:] != '.map':
            filename = dump_file + '.map'
        f_out = open(filename, 'w')
        tag = self.rigid_count
        for cage in self.cages:
            string = "cage "
            for index in range(len(cage.positions[:cage.num_connected])):
                string += str(tag)
                string += " "
                tag += 1
            string += "\n"
            f_out.write(string)
            string = "graft_sites "
            for index in cage.graft_sites:
                string += str(self.rigid_count + index)
                string += " "
            string += "\n"
            f_out.write(string)
            for index in range(cage.num_connected, len(cage.positions)):
                if cage.types[index][0] == "P":
                    qpi_list.append(index + self.rigid_count)
                else:
                    qmi_list.append(index + self.rigid_count)
            for chain in cage.grafted:
                if chain is not None:
                    string = "grafted_polymer "
                    for ind, name in enumerate(chain.sequence):
                        string += name
                        string += " "
                        for index in chain.monomer_indexes[ind]:
                            try:
                                chain.type[index][-1] != "i"
                            except IndexError:
                                print(index, chain.type, len(chain.type))
                            if chain.type[index][-1] != "i":
                                string += str(index + tag)
                                string += " "
                            else:
                                if chain.type[index][0] == "P":
                                    qpi_list.append(index + tag)
                                else:
                                    qmi_list.append(index + tag)
                    string += "\n"
                    f_out.write(string)
        for chain in self.chains:
            string = "free_polymer "
            for ind, name in enumerate(chain.sequence):
                string += name
                string += " "
                for index in chain.monomer_indexes[ind]:
                    try:
                        chain.type[index][-1] != "i"
                    except IndexError:
                        print(index, chain.type, len(chain.type))
                    if chain.type[index][-1] != "i":
                        string += str(index + tag)
                        string += " "
                    else:
                        if chain.type[index][0] == "P":
                            qpi_list.append(index+tag)
                        else:
                            qmi_list.append(index+tag)
            string += "\n"
            f_out.write(string)
            tag += chain.num_beads
        string = "qmi "
        for index in qmi_list:
            string += str(index)
            string += " "
        string += "\n"
        f_out.write(string)
        string = "qpi "
        for index in qpi_list:
            string += str(index)
            string += " "
        string += "\n"
        f_out.write(string)

        f_out.close()

    def distance(self, point1, point2):

        one = cp.deepcopy(point1)
        two = cp.deepcopy(point2)
        sub = [0, 0, 0]
        for i in range(3):
            diff = np.abs(one[i] - two[i])
            if diff > np.diag(self.cell_matrix)[i]/2:
                diff = np.diag(self.cell_matrix)[i] - diff
            sub[i] = diff
        return la.norm(sub)

    def clash(self, point, cage_index, dist):

        cen = self.cages[cage_index].center
        if self.distance(cen, point) < dist:
            #print("cage cleas")
            return True
        return False

    def clashes(self, point, radius):
        for cage_index in range(len(self.cages)):
            if self.clash(point, cage_index, radius):
                return True
        return False

    def place_poly(self, poly, starting_point, radius, check_start=False, mon_length=0.5):
        if check_start:
            if self.clashes(starting_point, radius):
                return False
        for index in range(len(poly.monomer_indexes)):
            if index == 0:
                start = starting_point
            else:
                start = poly.position[poly.monomer_indexes[index-1][0]]
            for ind, bead_index in enumerate(poly.monomer_indexes[index]):
                if ind != 0:
                    start = poly.position[poly.monomer_indexes[index][ind - 1]]
                poly.position[bead_index] = np.add(start, self.random_vector(length=mon_length))
                mon_count = 0
                while self.clashes(poly.position[bead_index], radius) or self.intrachain_clash(poly, bead_index, mon_length*1.5):
                    #print(bead_index, ind)
                    mon_count += 1
                    #print(mon_count)
                    if mon_count > 1000:
                        return False
                    poly.position[bead_index] = np.add(start, self.random_vector(length=mon_length))
                #print("mon index", index)
        return True

    def place_poly_on_positions(self, poly, mon_length=1.0):

        qindex, starting_point = self.random_poly_point()
        #print(self.poly_points)
        #print(starting_point)
        #self.poly_points.pop(qindex)
        #temp = [starting_point]
        temp = []

        for index in range(len(poly.monomer_indexes)):
            if index == 0:
                start = starting_point
            else:
                start = poly.position[poly.monomer_indexes[index-1][0]]
            for ind, bead_index in enumerate(poly.monomer_indexes[index]):
                if ind != 0:
                    start = poly.position[poly.monomer_indexes[index][ind - 1]]
                if index == 0 and ind ==0:
                    rindex, next = qindex, start
                else:
                    rindex, next = self.adjacent_poly_point(start, mon_length=mon_length)
                if np.sum(next) > -2999:
                    temp.append(next)
                    poly.position[bead_index] = next
                    self.poly_points.pop(rindex)
                else:
                    self.poly_points.extend(temp)
                    return False
        return True


    def intrachain_clash(self, poly, bead_index, dist):

        for i in range(bead_index - 2, -1,-1):
            #print(bead_index, i)
            if self.distance(poly.position[bead_index], poly.position[i]) < dist:
                #print("intra clash", self.distance(poly.position[bead_index], poly.position[i]))
                return True
        return False

    def interchain_clashes(self, poly, polys, cut=1):

        #print("called")
        for p in polys:
            if self.interchain_clash(p, poly, cut=cut):
                #print("inter clash")
                return True
        return False

    def interchain_clash(self, poly1, poly2, cut=0.5):

        #print("also called")
        for p1 in poly1.position:
            for p2 in poly2.position:
                #print(la.norm(np.subtract(p1,p2)))
                if self.distance(p1, p2) < cut:
                    #print("inter")
                    return True
        return False

    def random_vector(self, length=1):
        direction = np.multiply(np.subtract(np.random.rand(3), .5), 2)
        unit = np.divide(direction, la.norm(direction))
        return np.multiply(unit, length)

    def random_point(self):
        frac_pos = np.random.rand(3)
        vectors = np.transpose(np.multiply(frac_pos, self.cell_matrix))
        pos = np.sum(vectors, axis=0)
        point_bottom = np.divide([np.sum(self.cell_matrix[i]) for i in range(3)], -2)
        return np.add(pos, point_bottom)

    def random_poly_point(self):

        qindex = np.random.randint(len(self.poly_points))
        return qindex, self.poly_points[qindex]

    def adjacent_poly_point(self, point, mon_length=1.0):

        #print("go")
        tried = []
        dir = list(range(3))

        #np.random.shuffle(dir)
        for i in dir:
            updown = [-1, 1]
            #np.random.shuffle(updown)
            for j in updown:
                thing = [0, 0, 0]
                thing[i] = j * mon_length
                adj = np.add(point, thing)
                #print(point, adj)
                #print(self.poly_points[-1], type(self.poly_points))
                vals = [ int(np.allclose(adj, ppoint)) for ppoint in self.poly_points]
                if np.sum(vals) == 1:
                    return vals.index(1), adj
                else:
                    tried.append(adj)

        return None, [-1000,-1000,-1000]

    def add_chains(self, graft=True):

        grafts = np.sum([len(cage.graft_sites) for cage in self.cages])
        current_cage = 0
        current_graft = 0
        grafts_added = 0
        radius = self.cages[0].rigid_max_radius() + .5
        for ind, chain in enumerate(self.chains):
            if ind < grafts and graft:
                #print(ind, current_cage, current_graft, grafts_added, self.cages[current_cage].graft_sites)
                #print("we grafted these")
                starting_point = self.cages[current_cage].positions[self.cages[current_cage].graft_sites[current_graft]]
                self.place_poly(chain, starting_point, radius, check_start=False, mon_length=self.mon_length)
                chain.position = np.subtract(chain.position, starting_point)
                self.cages[current_cage].graft(chain, current_graft)
                current_graft += 1
                grafts_added += 1
                if current_graft == len(self.cages[current_cage].graft_sites):
                    current_cage += 1
                    current_graft = 0
            else:
                #print("freedom")
                if True:
                    starting_point = self.random_point()
                    while (not self.place_poly(chain, starting_point, radius, check_start=True, mon_length=self.mon_length)):
                        starting_point = self.random_point()
                #while (not self.place_poly(chain, starting_point, radius, check_start=True)) or self.interchain_clashes(chain, self.chains[:ind]) :
                    #starting_point = self.random_point()
                else:
                    while not self.place_poly_on_positions(chain, mon_length=self.mon_length):
                        keep_going = True
                        print("looping", len(self.poly_points))
                        print(ind, len(self.poly_points))
                print(ind)
        self.chains = self.chains[grafts_added:]

    def dump_xyz(self, title):

        num = np.sum([len(chain.position) for chain in self.chains]) + np.sum([len(cage.positions) for cage in self.cages])

        f = open(title + ".xyz", "w")
        f.write(str(num))
        f.write("\n\n")
        count = 0
        mon_count = 0

        for ind, cage in enumerate(self.cages):
            for mon in range(len(cage.positions)):
                count += 1
                mon_count += 1
                s = "%5s%8.3f%8.3f%8.3f\n" % (
                    cage.types[mon], cage.positions[mon][0], cage.positions[mon][1], cage.positions[mon][2])
                f.write(s)
                mon_count = 0

        for ind, chain in enumerate(self.chains):
            for mon in range(chain.num_beads):
                count += 1
                mon_count += 1
                s = "%5s%8.3f%8.3f%8.3f\n" % (
                    chain.type[mon], chain.position[mon][0], chain.position[mon][1], chain.position[mon][2])
                f.write(s)
            mon_count = 0

        f.write(str(self.box_length) + " " + str(self.box_length) + " " + str(self.box_length))
        f.close()

    def add_calcium(self, n):

        for i in range(int(n)):
            self.chains.append(SequentialPolymer(["CalciumMonomer"]))