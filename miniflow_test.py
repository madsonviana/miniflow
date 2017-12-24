# -*- coding: utf-8 -*-

import unittest
from nodes.node import Node
from nodes.add import Add
from nodes.input import Input
class TestNodes(unittest.TestCase):
    """
    Classe de teste para implementação dos nós da rede neural
    """

    def test_node_interface(self):
        """
        Testa se não tem implementação de forward na classe Node
        """
        node = Node()
        with self.assertRaises(NotImplementedError):
            node.forward()

    def test_add_node(self):
        """
        Testa o nó de adição
        """
        input1 = Input()
        input1.forward(1)

        input2 = Input()
        input2.forward(2)

        add = Add(input1, input2)
        add.forward()

        self.assertEqual(3, add.value)
