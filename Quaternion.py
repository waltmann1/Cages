"""
:module: Quaternion
:platform: Unix, Windows
:synopsis: A Quaternion operations class

.. moduleauthor:: Curt Waltmann <waltmann@iastate.edu> May 2017
"""

import numpy as np
from numpy import linalg as la


class Quaternion(object):
    """
        quaternion class
    """
    def __init__(self, orientation, hoomd=False):
        """

        :param hoomd_orientation: quaternion in hoomd format [real,x,y,z]
        """

        orientation = list(orientation)
        if (hoomd):
            self.q = np.array((orientation[1], orientation[2], orientation[3], orientation[0]), dtype=np.float64)
        else:
            self.q = np.array((orientation[0], orientation[1], orientation[2], orientation[3]), dtype=np.float64)

    def multiply(self, quaternion2):
        """

        :param quaternion2: quaternion to multipply this quaternion by
        :return: another quaternion which is the result of the multiplication
        """
        x0, y0, z0, w0 = quaternion2.q
        x1, y1, z1, w1 = self.q
        new = np.array((
            x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
            -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
            x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0,
            -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0), dtype=np.float64)
        return Quaternion(new)

    def conjugate(self):
        """

        :return: the quaternion's conjugate as a quaternion object
        """
        return Quaternion(np.array((-self.q[0], -self.q[1], -self.q[2], self.q[3]), dtype=np.float64))

    def orient(self, vector):
        """
        :param vector: the vector to be rotated [x, y, z]
        :return: the rotated vector [x, y, z]
        """
        q2 = [vector[0], vector[1], vector[2], 0.0]
        v_quat = Quaternion(q2)
        return self.multiply(v_quat).multiply(self.conjugate()).q[:3]

    def inverse(self):
        """
        :return: the inverse quaternion
        """
        q0, q1, q2, q3 = self.q
        bottom = q0 ** 2 + q1 ** 2 + q2 ** 2 + q3 ** 2
        q4 = np.divide([-q0, -q1, -q2, q3], bottom)
        return Quaternion(q4)

    def de_orient(self, position):
        """
        :param position: position vector to be deoriented [x, y, z]
        :return: the "unrotated" position
        """

        return self.inverse().orient(position)

class QuaternionBetween(Quaternion):
    """
    calculates Quaternion between 2 Vectors
    """

    def __init__(self, vector1, vector2, hoomd=False):
        """

        :param vector1: vector the quaternion goes from
        :param vector2: vector the quaternion goes to
        :param hoomd: set to true if you want hoomd style Quaternion
        """

        cross = np.cross(vector1, vector2)
        w = la.norm(vector1) * la.norm(vector2) + np.dot(vector1, vector2)
        length = la.norm([cross[0], cross[1], cross[2], w])
        unit = [cross[0], cross[1], cross[2], w] / length

        super(QuaternionBetween, self).__init__(unit, hoomd=hoomd)
