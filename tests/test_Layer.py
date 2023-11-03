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


def test_get_child():
    layer1 = Layer(LayerTypes.INPUT)
    layer2 = Layer(LayerTypes.PERC, left_layer=layer1)
    p1 = Perceptron(ActivationFunctions.SIGMOID_BIPOLAR)
    p2 = Perceptron(ActivationFunctions.SOFTPLUS)
    layer2.add_child(p1)
    layer2.add_child(p2)
    assert layer2.get_child(0) == p1
    assert layer2.get_child(1) == p2
    with pytest.raises(ValueError):
        layer2.get_child(2)
    with pytest.raises(ValueError):
        layer2.get_child(-1)
    with pytest.raises(ValueError):
        layer2.get_child(3)


def test_get_children():
    layer1 = Layer(LayerTypes.INPUT)
    layer2 = Layer(LayerTypes.PERC, left_layer=layer1)
    p1 = Perceptron(ActivationFunctions.SIGMOID_BIPOLAR)
    p2 = Perceptron(ActivationFunctions.SOFTPLUS)
    layer2.add_child(p1)
    layer2.add_child(p2)
    result = layer2.get_children()
    assert isinstance(result, list)
    assert len(result) == 2
    assert result[0] == p1
    assert result[1] == p2


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
    layer1.set_id(0)
    layer1.add_children([NetworkInput(), NetworkInput()])
    layer1.get_child(0).output = -1.3
    layer1.get_child(1).output = 0.5
    layer2 = Layer(LayerTypes.PERC, left_layer=layer1)
    layer2.set_id(1)
    layer2.add_children([Perceptron(ActivationFunctions.IDENTITY),
                         Perceptron(ActivationFunctions.RELU)])
    layer2.set_children_weights([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert layer2.get_sums() == [-0.1, 0.5]


def test_set_children_functions():
    layer1 = Layer(LayerTypes.INPUT, None, None)
    layer2 = Layer(LayerTypes.PERC, layer1, None)
    layer2.add_children([Perceptron(ActivationFunctions.IDENTITY),
                         Perceptron(ActivationFunctions.IDENTITY)])
    layer2.set_children_functions(ActivationFunctions.SOFTPLUS)
    assert layer2.get_child(0).activation_function == ActivationFunctions.SOFTPLUS
    assert layer2.get_child(1).activation_function == ActivationFunctions.SOFTPLUS
    with pytest.raises(ValueError):
        layer2.set_children_functions([ActivationFunctions.SIGMOID_BIPOLAR])
    with pytest.raises(ValueError):
        layer2.set_children_functions('text')


def test_set_child_weights():
    layer1 = Layer(LayerTypes.INPUT)
    layer2 = Layer(LayerTypes.PERC, layer1)
    layer1.add_children([NetworkInput(), NetworkInput()])
    layer2.add_children([Perceptron(ActivationFunctions.SIGMOID_BIPOLAR),
                         Perceptron(ActivationFunctions.SOFTPLUS)])
    assert layer2.get_child(0).weights == [0.0, 0.0, 0.0]
    layer2.set_child_weights(0, [1.0, 2.0, 3.0])
    assert layer2.get_child(0).weights == [1.0, 2.0, 3.0]
    with pytest.raises(ValueError):
        layer2.set_child_weights(13, [1.0, 2.0])
    with pytest.raises(ValueError):
        layer2.set_child_weights(0, [1.0, 2.0])


def test_set_children_weights():
    layer1 = Layer(LayerTypes.INPUT, None, None)
    layer2 = Layer(LayerTypes.PERC, layer1, None)
    layer1.add_children([NetworkInput(), NetworkInput()])
    layer2.add_children([Perceptron(ActivationFunctions.SIGMOID_BIPOLAR),
                         Perceptron(ActivationFunctions.SOFTPLUS)])
    assert layer2.get_child(0).weights == [0.0, 0.0, 0.0]
    assert layer2.get_child(1).weights == [0.0, 0.0, 0.0]
    layer2.set_children_weights([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    assert layer2.get_child(0).weights == [1.0, 2.0, 3.0]
    assert layer2.get_child(1).weights == [4.0, 5.0, 6.0]
    with pytest.raises(ValueError):
        layer2.set_children_weights([[1.0, 2.0, 3.0], [4.0, 5.0]])
    with pytest.raises(ValueError):
        layer2.set_children_weights([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0]])


def test_set_children_functions_by_list():
    layer1 = Layer(LayerTypes.INPUT, None, None)
    layer2 = Layer(LayerTypes.PERC, layer1, None)
    layer2.add_children([Perceptron(ActivationFunctions.IDENTITY),
                         Perceptron(ActivationFunctions.IDENTITY)])
    layer2.set_children_functions_by_list([ActivationFunctions.SOFTPLUS, ActivationFunctions.SOFTPLUS])
    assert layer2.children[0].activation_function == ActivationFunctions.SOFTPLUS
    assert layer2.children[1].activation_function == ActivationFunctions.SOFTPLUS
    with pytest.raises(ValueError):
        layer2.set_children_functions_by_list([ActivationFunctions.SIGMOID_BIPOLAR])


def test_set_child_function():
    layer1 = Layer(LayerTypes.INPUT, None, None)
    layer2 = Layer(LayerTypes.PERC, layer1, None)
    layer2.add_children([Perceptron(ActivationFunctions.IDENTITY),
                         Perceptron(ActivationFunctions.IDENTITY)])
    layer2.set_child_function(0, ActivationFunctions.SOFTPLUS)
    assert layer2.children[0].activation_function == ActivationFunctions.SOFTPLUS
    assert layer2.children[1].activation_function == ActivationFunctions.IDENTITY
    with pytest.raises(ValueError):
        layer2.set_child_function(13, ActivationFunctions.SOFTPLUS)


def test_get_children_functions():
    layer1 = Layer(LayerTypes.INPUT, None, None)
    layer2 = Layer(LayerTypes.PERC, layer1, None)
    layer2.add_children([Perceptron(ActivationFunctions.IDENTITY),
                         Perceptron(ActivationFunctions.SIGMOID_BIPOLAR)])
    assert layer2.get_children_functions() == [ActivationFunctions.IDENTITY, ActivationFunctions.SIGMOID_BIPOLAR]


def test_validate():
    layer1 = Layer(LayerTypes.INPUT, None, None)
    layer2 = Layer(LayerTypes.PERC, layer1, None)
    layer3 = Layer(LayerTypes.PERC, layer2, None)
    layer4 = Layer(LayerTypes.OUTPUT, layer3, None)
    with pytest.raises(ValueError):
        layer1.validate()
    with pytest.raises(ValueError):
        layer3.validate()
    layer4.validate()


def test_debug():
    layer4 = Layer(LayerTypes.OUTPUT)
    layer4.add_children([NetworkOutput(), NetworkOutput()])
    layer1 = Layer(LayerTypes.INPUT, None, None)
    layer2 = Layer(LayerTypes.PERC, layer1, None)
    layer1.add_children([NetworkInput(), NetworkInput()])
    layer2.add_children([Perceptron(ActivationFunctions.SIGMOID_BIPOLAR),
                         Perceptron(ActivationFunctions.SOFTPLUS)])
    layer1.debug_indepth()
    layer2.debug_indepth()
    layer4.debug_indepth()


def test_connect():
    layer1 = Layer(LayerTypes.INPUT)
    layer1.set_id(0)
    layer2 = Layer(LayerTypes.PERC)
    layer2.set_id(1)
    layer3 = Layer(LayerTypes.OUTPUT)
    layer3.set_id(2)
    layer1.add_children([NetworkInput(), NetworkInput()])
    layer2.add_children([Perceptron(ActivationFunctions.SIGMOID_BIPOLAR),
                         Perceptron(ActivationFunctions.SOFTPLUS)])
    layer3.add_children([NetworkOutput(), NetworkOutput()])
    layer2.connect(layer1, layer3)
    assert layer2.left_layer == layer1
    assert layer2.right_layer == layer3
    assert layer2.get_child(0).left_neightbours[1:] == layer1.children
    assert layer2.get_child(1).left_neightbours[1:] == layer1.children
    assert layer2.get_child(0).output == 0.0
    assert layer2.get_child(1).output == 0.0
    assert layer2.get_child(0).weights == [0.0, 0.0, 0.0]
    assert layer2.get_child(1).weights == [0.0, 0.0, 0.0]
    assert layer2.get_child(0).activation_function == ActivationFunctions.SIGMOID_BIPOLAR
    assert layer2.get_child(1).activation_function == ActivationFunctions.SOFTPLUS
    assert layer2.get_child(0).id == 'P/1/1'
    assert layer2.get_child(1).id == 'P/1/2'
    assert layer1.get_child(0).id == 'I/0/1'
    assert layer1.get_child(1).id == 'I/0/2'
    assert layer3.get_child(0).id == 'O/2/1'
    assert layer3.get_child(1).id == 'O/2/2'
    with pytest.raises(ValueError):
        layer2.connect(layer1, layer3)
    with pytest.raises(ValueError):
        layer2.connect(layer3, layer1)
    with pytest.raises(ValueError):
        layer2.connect(layer1, layer2)
    with pytest.raises(ValueError):
        layer2.connect(layer2, layer1)
    with pytest.raises(ValueError):
        layer2.connect(layer3, layer2)
    with pytest.raises(ValueError):
        layer2.connect(layer2, layer3)
    with pytest.raises(ValueError):
        layer2.connect(layer2, layer2)


def test_get_dict():
    layer1 = Layer(LayerTypes.INPUT, None, None)
    layer2 = Layer(LayerTypes.PERC, layer1, None)
    layer2.set_id(1)
    layer2.add_children([Perceptron(ActivationFunctions.IDENTITY),
                         Perceptron(ActivationFunctions.IDENTITY)])
    activation_function = ActivationFunctions.SIGMOID_BIPOLAR
    layer2.set_children_functions_by_list([activation_function, activation_function])
    expected_dict = {
        'id': 'L/P/1/2',
        'layer-type': LayerTypes.PERC,
        'left-layer': layer1,
        'right_layer': None,
        'children': layer2.children
    }
    assert layer2.get_dict() == expected_dict


def test_repr():
    layer1 = Layer(LayerTypes.INPUT, None, None)
    assert layer1.__repr__() == 'L/?/?/'
    layer1.set_id(0)
    assert layer1.__repr__() == 'L/I/0/0'
    layer1.add_children([NetworkInput(), NetworkInput()])
    assert layer1.__repr__() == 'L/I/0/2'
    layer2 = Layer(LayerTypes.PERC, layer1, None)
    assert layer2.__repr__() == 'L/?/?/'
    layer2.set_id(1)
    assert layer2.__repr__() == 'L/P/1/0'
    layer2.add_child(Perceptron(ActivationFunctions.SIGMOID_BIPOLAR))
    assert layer2.__repr__() == 'L/P/1/1'
    layer2.add_child(Perceptron(ActivationFunctions.SOFTPLUS))
    layer3 = Layer(LayerTypes.OUTPUT, layer2, None)
    assert layer3.__repr__() == 'L/?/?/'
    layer3.set_id(2)
    assert layer3.__repr__() == 'L/O/2/0'
    layer3.add_child(NetworkOutput())
    assert layer3.__repr__() == 'L/O/2/1'
    layer3.add_child(NetworkOutput())
    assert layer3.__repr__() == 'L/O/2/2'