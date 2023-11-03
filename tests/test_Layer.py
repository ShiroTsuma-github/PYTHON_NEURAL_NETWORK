import os
import pytest
import sys
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

from src.Layer import Layer, LayerTypes, ActivationFunctions, Perceptron, NetworkInput, NetworkOutput    # noqa: E402


def test_add_child():
    layer1 = Layer(LayerTypes.INPUT)
    layer2 = Layer(LayerTypes.PERC, left_layer=layer1)
    layer2.set_id(1)
    layer1.set_id(0)
    layer3 = Layer(LayerTypes.OUTPUT, left_layer=layer2)
    layer2.add_child(Perceptron(ActivationFunctions.SIGMOID_BIPOLAR))
    assert layer2.get_child_count() == 1
    layer2.add_child(Perceptron(ActivationFunctions.SOFTPLUS))
    assert layer2.get_child_count() == 2
    with pytest.raises(ValueError):
        layer1.add_child(Perceptron(ActivationFunctions.SIGMOID_BIPOLAR))
    with pytest.raises(ValueError):
        layer3.add_child(Perceptron(ActivationFunctions.SIGMOID_BIPOLAR))
    with pytest.raises(ValueError):
        layer2.add_child(NetworkInput())
    with pytest.raises(ValueError):
        layer2.add_child(NetworkOutput())
    layer1.add_child(NetworkInput())
    assert layer2.children[0].activation_function == ActivationFunctions.SIGMOID_BIPOLAR
    assert layer2.children[1].activation_function == ActivationFunctions.SOFTPLUS
    assert layer2.children[0].id == 'P/1/1'
    assert layer2.children[1].id == 'P/1/2'
    assert layer1.children[0].id == 'I/0/1'


def test_add_children():
    layer1 = Layer(LayerTypes.INPUT)
    layer2 = Layer(LayerTypes.PERC, left_layer=layer1)
    layer2.set_id(1)
    layer1.set_id(0)
    layer3 = Layer(LayerTypes.OUTPUT, left_layer=layer2)
    layer2.add_children([Perceptron(ActivationFunctions.SIGMOID_BIPOLAR), Perceptron(ActivationFunctions.SOFTPLUS)])
    assert layer2.get_child_count() == 2
    with pytest.raises(ValueError):
        layer1.add_children([Perceptron(ActivationFunctions.SIGMOID_BIPOLAR), Perceptron(ActivationFunctions.SOFTPLUS)])
    with pytest.raises(ValueError):
        layer3.add_children([Perceptron(ActivationFunctions.SIGMOID_BIPOLAR), Perceptron(ActivationFunctions.SOFTPLUS)])
    with pytest.raises(ValueError):
        layer2.add_children([NetworkInput(), NetworkOutput()])
    layer1.add_children([NetworkInput(), NetworkInput()])
    with pytest.raises(ValueError):
        layer2.add_children([NetworkInput(), NetworkOutput()])
    with pytest.raises(ValueError):
        layer2.add_children(Perceptron(ActivationFunctions.SIGMOID_BIPOLAR))


def test_get_child_count():
    layer1 = Layer(LayerTypes.INPUT)
    layer2 = Layer(LayerTypes.PERC, left_layer=layer1)
    layer2.add_child(Perceptron(ActivationFunctions.SIGMOID_BIPOLAR))
    assert layer2.get_child_count() == 1
    layer2.add_child(Perceptron(ActivationFunctions.SOFTPLUS))
    assert layer2.get_child_count() == 2
    layer2.add_child(Perceptron(ActivationFunctions.SOFTPLUS))


def test_set_children_ids():
    layer1 = Layer(LayerTypes.INPUT)
    layer2 = Layer(LayerTypes.PERC, left_layer=layer1)
    layer2.add_child(Perceptron(ActivationFunctions.SIGMOID_BIPOLAR))
    layer2.add_child(Perceptron(ActivationFunctions.SOFTPLUS))
    layer2.set_id(1)
    layer1.set_id(0)
    layer2.set_children_ids()
    assert layer2.children[0].id == 'P/1/1'
    assert layer2.children[1].id == 'P/1/2'


def test_get_sums():
    layer1 = Layer(LayerTypes.INPUT)
    layer1.add_children([NetworkInput(), NetworkInput()])
    layer1.get_child(0).output = -1.3
    layer1.get_child(1).output = 0.5
    layer2 = Layer(LayerTypes.PERC, left_layer=layer1)
    layer2.children = [Perceptron(ActivationFunctions.IDENTITY),
                       Perceptron(ActivationFunctions.RELU)]
    layer2.set_children_weights[[1], [0.3]]
    assert layer2.get_sums() == [0.0, 0.0]

def test_set_children_functions():
    layer1 = Layer(LayerTypes.INPUT, None, None)
    layer2 = Layer(LayerTypes.PERC, layer1, None)
    layer2.children = [Perceptron(2), Perceptron(2)]
    activation_function = ActivationFunctions.SIGMOID
    layer2.set_children_functions(activation_function)
    assert layer2.children[0].activation_function == activation_function
    assert layer2.children[1].activation_function == activation_function

def test_set_child_weights():
    layer1 = Layer(LayerTypes.INPUT, None, None)
    layer2 = Layer(LayerTypes.PERC, layer1, None)
    layer3 = Layer(LayerTypes.OUTPUT, layer2, None)
    layer4 = Layer(LayerTypes.PERC, layer2, layer3)
    layer4.children = [Perceptron(2), Perceptron(2)]
    layer4.left_layer.children = [Perceptron(2), Perceptron(2)]
    layer4.set_child_weights(0, [1.0, 2.0, 3.0])
    assert layer4.children[0].weights == [1.0, 2.0, 3.0]

def test_set_children_weights():
    layer1 = Layer(LayerTypes.INPUT, None, None)
    layer2 = Layer(LayerTypes.PERC, layer1, None)
    layer3 = Layer(LayerTypes.OUTPUT, layer2, None)
    layer4 = Layer(LayerTypes.PERC, layer2, layer3)
    layer4.children = [Perceptron(2), Perceptron(2)]
    layer4.left_layer.children = [Perceptron(2), Perceptron(2)]
    layer4.set_children_weights([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert layer4.children[0].weights == [1.0, 2.0, 3.0]
    assert layer4.children[1].weights == [4.0, 5.0, 6.0]

def test_set_children_functions_by_list():
    layer1 = Layer(LayerTypes.INPUT, None, None)
    layer2 = Layer(LayerTypes.PERC, layer1, None)
    layer2.children = [Perceptron(2), Perceptron(2)]
    activation_function = ActivationFunctions.SIGMOID
    layer2.set_children_functions_by_list([activation_function, activation_function])
    assert layer2.children[0].activation_function == activation_function
    assert layer2.children[1].activation_function == activation_function

def test_set_child_function():
    layer1 = Layer(LayerTypes.INPUT, None, None)
    layer2 = Layer(LayerTypes.PERC, layer1, None)
    layer2.children = [Perceptron(2), Perceptron(2)]
    activation_function = ActivationFunctions.SIGMOID
    layer2.set_child_function(0, activation_function)
    assert layer2.children[0].activation_function == activation_function

def test_get_children_functions():
    layer1 = Layer(LayerTypes.INPUT, None, None)
    layer2 = Layer(LayerTypes.PERC, layer1, None)
    layer2.children = [Perceptron(2), Perceptron(2)]
    activation_function = ActivationFunctions.SIGMOID
    layer2.set_children_functions_by_list([activation_function, activation_function])
    assert layer2.get_children_functions() == [activation_function, activation_function]

def test_validate():
    layer1 = Layer(LayerTypes.INPUT, None, None)
    layer2 = Layer(LayerTypes.PERC, layer1, None)
    layer3 = Layer(LayerTypes.OUTPUT, layer2, None)
    layer4 = Layer(LayerTypes.PERC, layer2, layer3)
    with pytest.raises(ValueError):
        layer1.validate()
    with pytest.raises(ValueError):
        layer3.validate()
    with pytest.raises(ValueError):
        layer4.validate()

def test_get_dict():
    layer1 = Layer(LayerTypes.INPUT, None, None)
    layer2 = Layer(LayerTypes.PERC, layer1, None)
    layer2.children = [Perceptron(2), Perceptron(2)]
    activation_function = ActivationFunctions.SIGMOID
    layer2.set_children_functions_by_list([activation_function, activation_function])
    expected_dict = {
        'id': 'L/PERC/2/1',
        'layer-type': LayerTypes.PERC,
        'left-layer': layer1,
        'right_layer': None,
        'children': layer2.children
    }
    assert layer2.get_dict() == expected_dict

def test_repr():
    layer1 = Layer(LayerTypes.INPUT, None, None)
    layer2 = Layer(LayerTypes.PERC, layer1, None)
    assert str(layer2) == 'L/PERC/2'