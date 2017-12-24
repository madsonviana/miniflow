# -*- coding: utf-8 -*-
from nodes.node import Node
from functools import reduce

class Mul(Node):
    def __init__(self, x, y):
        Node.__init__(self, [x, y])
    
    def forward(self):
        self.value = reduce(lambda x, y: x * y, map(lambda x: x.value, self.inbound_nodes))