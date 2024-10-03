from unittest import TestCase
import numpy as np
import random
import unittest

from rsa_backward.rsa_data import (
    generate_distributions,
    sample_scenario,
    SingleConfig,
    generate_target,
    compute_l_0,
    generate_messages,
    create_association_table,
    compute_listener_dist,
    compute_speaker_dist
)

P_S = [0.01138892, 0.07208168, 0.38734778, 0.52918161]

P_S_S = {
    'circle': np.array([0.20390963, 0.30684722, 0.48869424, 0.00054892]),
    'square': np.array([0.19019651, 0.09041291, 0.35102138, 0.3683692 ]),
    'rectangle': np.array([0.44661377, 0.12446258, 0.19717176, 0.2317519 ]),
    'ellipse': np.array([0.13843352, 0.47285487, 0.20493806, 0.18377355])}
P_C_S = {
    'circle': np.array([0.08782333, 0.04792849, 0.27419609, 0.20613712, 0.33604495,0.04787002]),
    'square': np.array([4.50813172e-01, 3.30793760e-03, 2.58017759e-01, 2.75479713e-01,4.15705020e-05, 1.23398484e-02]),
    'rectangle': np.array([0.23028159, 0.2795595 , 0.0606748 , 0.19543154, 0.18435741,0.04969516]),
    'ellipse': np.array([0.18970641, 0.20327017, 0.0654691 , 0.0741099 , 0.07882668, 0.38861775])}

class TestRSAGeneration(TestCase):

    def SetUp(self):
        random.seed(0)

    def test_sample_scenario(self):
        random.seed(0)
        configs = sample_scenario(2,P_S, P_S_S, P_C_S)
        expected = [
            SingleConfig(shape='ellipse', color='gray'),
            SingleConfig(shape='square', color='red'),
            SingleConfig(shape='square', color='red')]
        self.assertEqual(configs, expected)

    def test_generate_target_positive(self):
        configs = [
            SingleConfig(shape='ellipse', color='gray'),
            SingleConfig(shape='square', color='red'),
            SingleConfig(shape='square', color='red')]

        label = generate_target(
            configs)
        self.assertEqual(label,0)

    def test_generate_target_negative(self):
        configs = [
            SingleConfig(shape='ellipse', color='gray'),
            SingleConfig(shape='ellipse', color='gray'),
            SingleConfig(shape='square', color='red'),
            SingleConfig(shape='square', color='red')]

        label = generate_target(
            configs)
        self.assertEqual(label,None)

    def test_generate_message_no_cost_single(self):
        configs = [
            SingleConfig(shape='ellipse', color='gray'),
            SingleConfig(shape='square', color='gray'),
            SingleConfig(shape='square', color='red')]

        label = 1

        message = generate_messages(configs, label)
        self.assertEqual(message,('square', 'gray'))

    def test_generate_message_with_cost(self):
        configs = [
            SingleConfig(shape='ellipse', color='gray'),
            SingleConfig(shape='square', color='gray'),
            SingleConfig(shape='square', color='red')]

        label = 1

        message = generate_messages(configs, label, alpha=1)
        self.assertEqual(message,('square',))

    def test_generate_message_multi(self):
        configs = [
            SingleConfig(shape='ellipse', color='gray'),
            SingleConfig(shape='ellipse', color='red'),
            SingleConfig(shape='square', color='red'),
            SingleConfig(shape='square', color='gray')
        ]

        label = 1

        message = generate_messages(configs, label)
        self.assertEqual(message,('ellipse', 'red'))

    def test_generate_message_multi_cost(self):
        configs = [
            SingleConfig(shape='ellipse', color='gray'),
            SingleConfig(shape='ellipse', color='red'),
            SingleConfig(shape='square', color='red'),
            SingleConfig(shape='square', color='gray')
        ]

        label = 1

        message = generate_messages(configs, label, alpha=0.6, speaker_level=1)
        self.assertEqual(message, ('ellipse', 'red'))
        message = generate_messages(configs, label, alpha=0.6, speaker_level=2)
        self.assertEqual(message,('ellipse','red'))

        message = generate_messages(configs[:3], label, alpha=0.6, speaker_level=1)
        self.assertEqual(message, ('ellipse', 'red'))
        message = generate_messages(configs[:3], label, alpha=0.6, speaker_level=2)
        self.assertEqual(message, ('ellipse',))

    def test_create_association_table(self):
        configs = [
            SingleConfig(shape='ellipse', color='gray'),
            SingleConfig(shape='ellipse', color='red'),
        ]
        array = create_association_table(configs)
        expected = np.array([[0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]])
        self.assertTrue(np.array_equal(array, expected))

    def test_create_listener_dist(self):
        configs = [
            SingleConfig(shape='ellipse', color='gray'),
            SingleConfig(shape='ellipse', color='red'),
        ]
        base_array = create_association_table(configs)
        out_dist = compute_listener_dist(base_array)
        self.assertEqual(out_dist[0,3], 0.5)

    def test_create_speaker_dist(self):
        configs = [
            SingleConfig(shape='ellipse', color='gray'),
            SingleConfig(shape='square', color='gray'),
            SingleConfig(shape='square', color='red')]
        base_array = create_association_table(configs)
        listener_dist = compute_listener_dist(base_array)
        speaker_dist = compute_speaker_dist(listener_dist)







if __name__ == '__main__':
    unittest.main()