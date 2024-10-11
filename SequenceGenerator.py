from __future__ import division
import numpy as np

class SequenceGenerator(object):

    def __init__(self, num_chains, chain_length, mono_names, mono_probabilities):

        if len(mono_probabilities) != len(mono_probabilities):
            raise ValueError("monomer names and probabilities must have the same length")
        if np.sum(mono_probabilities) != 1:
            raise ValueError("Monomer probabilities must sum to 1")

        for thing in mono_names:
            try:
                imp = __import__(thing, fromlist=[''])
                mon = getattr(imp, thing)()
            except:
                raise ValueError(thing + "is not a Monomer class that exists")

        self.monomer_names = mono_names
        self.monomer_probabilities = mono_probabilities
        self.num_chains = num_chains
        self.chain_length = chain_length

    def get_sequences(self):

        mono_left = [int(self.num_chains * self.chain_length * self.monomer_probabilities[i])
                     for i in range(len(self.monomer_probabilities))]

        counter = 0
        while np.sum(mono_left) > self.num_chains * self.chain_length:
            mono_left[counter % len(self.monomer_names)] -= 1

        while np.sum(mono_left) < self.num_chains * self.chain_length:
            mono_left[counter % len(self.monomer_names)] += 1

        chains = [[] for j in range(self.num_chains)]

        for x in range(self.num_chains):
            for y in range(self.chain_length):
                rand = np.random.random_sample()
                done = False
                for i in range(len(self.monomer_names)):
                    if rand <= np.sum(self.monomer_probabilities[:i + 1]) and not done:
                        chains[x].append(self.monomer_names[i])
                        mono_left[i] -= 1
                        self.recalculate_probabilities(mono_left)
                        done = True
        return chains

    def recalculate_probabilities(self, mono_left):

        self.monomer_probabilities = [mono_left[i] / np.sum(mono_left) for i in range(len(mono_left))]