from __future__ import division
from AbstractCage import AbstractCage
import numpy as np
from numpy import linalg as la
from Quaternion import QuaternionBetween

class FerritinCage(AbstractCage):

    def __init__(self, four_dist, three_dist, uniform_total_charge=None, pqr="parse_pH6.5_SingleCage2_noraft.pqr", qxyz=None):

        super(FerritinCage, self).__init__()

        if qxyz is not None:
            f = open(qxyz)
            for line in f.readlines()[2:]:
                s = line.split()
                self.charges.append(float(s[0]))
                self.positions.append([float(s[1]), float(s[2]), float(s[3])])
                self.masses.append(1)
                self.types.append('N')
                self.images.append([0, 0, 0])
        else:
            points, total_faces = self.ferrtin_poitions(four_dist, three_dist)
            for ind, point in enumerate(points):
                self.positions.append(point)
                self.masses.append(1)
                self.types.append('N')
                self.charges.append(0)
                self.images.append([0,0,0])
                total_count = 0
                #for face in total_faces:
                #    face_count = 0
                #    for ind, point in enumerate(face):
                #        if point is not None:
                #           print(face_count)
                #           self.charges[total_count] = face_count
                #           total_count += 1
                #       face_count += 1


            pqr_pos, pqr_charges = self.read_pqr(pqr)

            quat = QuaternionBetween(np.divide([1, 1, 1], np.sqrt(3)), [1,0,0])
            self.align(quat)
            self.dump_xyz("mid1")


            self.assign_charges_vector(pqr_pos, pqr_charges)

            num = len(pqr_pos)

            f = open("pqr_mid1" + ".xyz", "w")
            f.write(str(num))
            f.write("\n\n")
            count = 0
            mon_count = 0

            for mon in range(num):
                count += 1
                mon_count += 1
                s = "%5s%8.3f%8.3f%8.3f\n" % (
                    "b", pqr_pos[mon][0], pqr_pos[mon][1], pqr_pos[mon][2])
                f.write(s)
                mon_count = 0
            f.close()

            quat = QuaternionBetween([1, 0, 0], np.divide([1, 1, 1], np.sqrt(3)))


            self.align(quat)
            self.dump_xyz("mid2")
            #quit()
        self.num_rigid = len(self.positions)
        self.num_connected = len(self.positions)
        self.moment = self.calculate_inertia_tensor()



    def ferrtin_poitions(self, four_dist, three_dist):

        points = []
        bcc_points = []
        sc_points = [[0,0,1], [1,0,0], [0,1,0], [0,0,-1], [-1,0,0], [0,-1,0]]
        sc_points = np.multiply(sc_points, four_dist)
        for i in range(-1,2,2):
            for j in range(-1, 2, 2):
                for k in range(-1, 2, 2):
                    bcc_points.append(np.multiply(three_dist, np.divide([i,j,k], la.norm([i,j,k]))))

        faces = []
        min_dist = four_dist + three_dist

        for point in sc_points:
            for point2 in sc_points:
                if not np.array_equal(point, point2) and not np.array_equal(point, -point2):
                    faces.append([point, point2])
        to_remove = []
        for ind in range(len(faces)-1, -1, -1):
            for i in range(ind -1, -1, -1):
                face_one = faces[ind]
                face_two = faces[i]
                if np.array_equal(face_one[0], face_two[0]) and np.array_equal(face_one[1], face_two[1]) or \
                        np.array_equal(face_one[1], face_two[0]) and np.array_equal(face_one[0], face_two[1]):
                    print("found")
                    to_remove.append(ind)

        print(to_remove, len(to_remove))
        for ind in to_remove:
            del faces[ind]

        min_dist = three_dist + four_dist
        for ind, face in enumerate(faces):
            to_append = []
            for point in bcc_points:
                dist1 = la.norm(np.subtract(face[0], point))
                dist2 = la.norm(np.subtract(face[1], point))
                if dist1 == dist2 and dist2 < min_dist:
                    min_dist = dist2
                    to_append = [point]
                elif dist1 == dist2 and dist2 == min_dist:
                    to_append.append(point)
            for a in to_append:
                faces[ind].append(a)

        bead_diameter = 1
        total_faces = []

        for ind, face in enumerate(faces):
            total_faces.append([])
            line1 = [face[0], face[2]]
            line2 = [face[3], face[1]]
            vec1 = np.subtract(face[0], face[2])
            vec2 = np.subtract(face[3], face[1])
            num_beads = int(np.ceil(la.norm(vec2)/bead_diameter))
            print(num_beads)
            for num in range(num_beads-1):
                line1.append(np.add(face[2], np.multiply((1 + num)/num_beads, vec1)))
                line2.append(np.add(face[1], np.multiply((1 + num) / num_beads, vec2)))

            #print(line1)
            #print(line2)
            #for h in range(len(line1)):
            #    print(la.norm(np.subtract(line1[h], line2[h])),np.subtract(line1[h], line2[h]) )
            #quit()
            lines = [[] for _ in range(len(line2))]
            for ind2 in range(len(line2)):
                lines[ind2].append(line1[ind2])
                lines[ind2].append(line2[ind2])
                vec3 = np.subtract(lines[ind2][0], lines[ind2][1])
                num_beads_2 = int(np.ceil(la.norm(vec3) / bead_diameter))
                for num in range(num_beads_2 - 1):
                    lines[ind2].append(np.add(lines[ind2][1], np.multiply((1 + num) / num_beads, vec3)))
                #print("lllllind",lines[ind2])
            #quit()
            for line in lines:
                for point in line:
                    total_faces[ind].append(point)


        #print(len(total_faces), len(total_faces[0]))
        #quit()
        points = []
        for ind3, face in enumerate(total_faces):
            for ind4, point in enumerate(face):
                there = False
                for point2 in points:
                    if np.array_equal(point, point2):
                        there = True
                if not there:
                    points.append(point)
                else:
                    total_faces[ind3][ind4] = None
        return points, total_faces









