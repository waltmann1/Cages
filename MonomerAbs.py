import numpy as np


class MonomerAbs(object):

    def __init__(self, with_ion=False):
        self.name = "MonomerAbs"
        self.mass = 1
        self.length = 1

    def add_to_polymer(self, p, spot, up=True):

        print("I don't ahve to do anything")