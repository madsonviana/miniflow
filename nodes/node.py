# -*- coding: utf-8 -*-
class Node(object):
    def __init__(self, inbound_nodes=[]):
        self.inbound_nodes = inbound_nodes
        self.outbound_nodes = []
        self.gradients = []

        for n in self.inbound_nodes:
            n.outbound_nodes.append(self)

        self.value = None

    def forward(self):
        raise NotImplementedError

    def backward(self):
        raise NotImplementedError
