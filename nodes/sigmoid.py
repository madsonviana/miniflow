# -*- coding: utf-8 -*-
from nodes.node import Node
import numpy as np

class Sigmoid(Node):
    def __init__(self, node):
        Node.__init__(self, [node])

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def forward(self):
        self.value = self._sigmoid(self.inbound_nodes[0].value)