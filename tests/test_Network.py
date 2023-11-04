import os
import pytest
import sys
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

from src.NeuralNetwork import NeuralNetwork, ActivationFunctions, LayerTypes    # noqa: E402


def test_setup():
    nn = NeuralNetwork()
    nn.setup(2, 2)
    assert nn.layers[0].get_child_count() == 2
    assert nn.get_layer_by_index(1).get_child_count() == 0
    assert nn.get_layer_by_index(2).get_child_count() == 0
    assert nn.layers[3].get_child_count() == 0
    assert nn.layers[0].layer_type == LayerTypes.INPUT
    assert nn.layers[1].layer_type == LayerTypes.PERC
    assert nn.layers[2].layer_type == LayerTypes.PERC
    assert nn.layers[3].layer_type == LayerTypes.OUTPUT
    assert nn.layers[0].right_layer == nn.layers[1]
    assert nn.layers[1].right_layer == nn.layers[2]
    assert nn.layers[2].right_layer == nn.layers[3]
    assert nn.layers[3].right_layer is None
    assert nn.layers[0].left_layer is None
    assert nn.layers[1].left_layer == nn.layers[0]
    assert nn.layers[2].left_layer == nn.layers[1]
    assert nn.layers[3].left_layer == nn.layers[2]


def test_set_perceptrons_per_layer():
    nn = NeuralNetwork()
    nn.setup(2, 2)
    nn.set_perceptrons_per_layer([3, 4])
    assert nn.get_perceptron_count() == 7
    assert len(nn.layers[0].get_children()) == 2
    assert len(nn.layers[1].get_children()) == 3
    assert len(nn.layers[2].get_children()) == 4


def test_set_layer_activation_function():
    nn = NeuralNetwork()
    nn.setup(2, 2)
    nn.set_perceptrons_per_layer([3, 4])
    nn.set_layer_activation_function(2, ActivationFunctions.SIGMOID_BIPOLAR)
    assert nn.layers[2].get_children_functions() == [ActivationFunctions.SIGMOID_BIPOLAR] * 4
    nn.set_layer_activation_function(1, ActivationFunctions.IDENTITY)
    assert nn.layers[1].get_children_functions() == [ActivationFunctions.IDENTITY] * 3
    with pytest.raises(ValueError):
        nn.set_layer_activation_function(0, ActivationFunctions.SIGMOID_BIPOLAR)
    with pytest.raises(ValueError):
        nn.set_layer_activation_function(3, ActivationFunctions.SIGMOID_BIPOLAR)


def test_get_layer_by_index():
    nn = NeuralNetwork()
    nn.setup(2, 2)
    assert nn.get_layer_by_index(1) == nn.layers[1]
    assert nn.get_layer_by_index(2) == nn.layers[2]
    with pytest.raises(ValueError):
        nn.get_layer_by_index(0)
    with pytest.raises(ValueError):
        nn.get_layer_by_index(3)


def test_save_and_load_network():
    nn = NeuralNetwork()
    nn.setup(2, 3)
    nn.set_perceptrons_per_layer([3, 4, 7])
    nn.set_layer_activation_function(2, ActivationFunctions.SIGMOID_UNIPOLAR)
    nn.get_layer_by_index(1).set_children_functions_by_list([
        ActivationFunctions.RELU,
        ActivationFunctions.RELU_LEAKY,
        ActivationFunctions.RELU_PARAMETRIC
    ])
    nn.set_input_values([1, 2])
    nn.save_network('tests/network.nn')
    nn2 = NeuralNetwork()
    nn2.load_network('tests/network.nn')
    assert nn2.get_input_values() == [1, 2]
    assert nn2.get_layer_by_index(1).get_child_count() == 3
    assert nn2.get_layer_by_index(2).get_child_count() == 4
    assert nn2.get_layer_by_index(3).get_child_count() == 7
    assert nn2.get_layer_by_index(2).get_children_functions() == [ActivationFunctions.SIGMOID_UNIPOLAR] * 4
    assert nn2.get_layer_by_index(1).get_children_functions() == [
        ActivationFunctions.RELU,
        ActivationFunctions.RELU_LEAKY,
        ActivationFunctions.RELU_PARAMETRIC
    ]