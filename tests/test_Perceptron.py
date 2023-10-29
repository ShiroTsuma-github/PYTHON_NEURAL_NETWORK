import pytest
import os
import sys
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

from src.Perceptron import Perceptron   # noqa: E402
from src.NetworkEnums import ActivationFunctions   # noqa: E402


def test_incorrect_output():
    p = Perceptron(ActivationFunctions.IDENTITY)
    with pytest.raises(ValueError):
        p.output = "a"
    with pytest.raises(ValueError):
        p.output = [1, 2, 3]
    p.output = -0.3
    assert p.output == -0.3


def test_output():
    p = Perceptron(ActivationFunctions.IDENTITY)
    p.output = 0.3
    assert p.output == 0.3
    p.output = 1
    assert p.output == 1

    for i in range(100):
        p.output = i

    assert p.output == 99
    assert len(p.previous_outputs) == 50
    assert p.previous_outputs[0] == 50
    assert p.previous_outputs[49] == 99


def test_weights():
    p = Perceptron(ActivationFunctions.IDENTITY)
    p2 = Perceptron(ActivationFunctions.RELU)
    with pytest.raises(ValueError):
        p.weights = [1, 2, 3]

    p.weights = [1]
    assert p.weights == [1]
    p.set_neighbours([p2])
    p.weights = [1, 2]
    with pytest.raises(ValueError):
        p.weights = [1, 2, 3]
    assert p.weights == [1, 2]
    with pytest.raises(ValueError):
        p.weights = "a"
    with pytest.raises(ValueError):
        p.weights = 1
    # with pytest.raises(ValueError):
    #     p.weights = [1, "a"]
    p.weights = [1, 'a']
    # do ogarnięcia

    for i in range(100):
        p.weights = [i, i + 1]
    
    assert p.weights == [99, 100]
    assert len(p.previous_weights) == 50
    assert p.previous_weights[0] == [50, 51]


def test_get_output_identity():
    p1 = Perceptron(ActivationFunctions.IDENTITY)
    p2 = Perceptron(ActivationFunctions.IDENTITY)
    p3 = Perceptron(ActivationFunctions.IDENTITY)
    p1.set_neighbours([p2, p3])
    p1.weights = [1, 2, 3]
    p2.output = 2
    p3.output = 3
    # 1 + 2*2 + 3*3 = 14
    assert p1.get_output() == 14
    assert p1.get_output() == p1.calc_identity()


def test_get_output_relu():
    p1 = Perceptron(ActivationFunctions.RELU)
    p2 = Perceptron(ActivationFunctions.RELU)
    p3 = Perceptron(ActivationFunctions.RELU)
    p1.set_neighbours([p2, p3])
    p1.weights = [1, 2, 3]
    p2.output = -1
    p3.output = 3
    # 1 + 2*-1 + 3*3 = 8
    assert p1.get_output() == 8
    p2.output = 1
    p3.output = 0.6
    p1.weights = [-0.1, 0.7, 0.4]
    assert p1.get_output() == 0.84
    assert p1.get_output() == p1.calc_relu()


def test_get_output_sigmoid_bipolar():
    p1 = Perceptron(ActivationFunctions.SIGMOID_BIPOLAR)
    p2 = Perceptron(ActivationFunctions.SIGMOID_BIPOLAR)
    p3 = Perceptron(ActivationFunctions.SIGMOID_BIPOLAR)
    p1.set_neighbours([p2, p3])
    p1.weights = [2, 2, 3]
    p2.output = -1
    p3.output = 3
    # 1 + 2*-1 + 3*3 = 8
    assert p1.get_output() == -0.9997532108480275
    p2.output = 1
    p3.output = 1
    p1.weights = [-0.1, -0.7, 0.4]
    assert p1.get_output() == 0.197375320224904
    assert p1.get_output() == p1.calc_sigmoid_bipolar()


def test_get_output_sigmoid_unipolar():
    pass


def test_get_output_softplus():
    pass


def test_get_output_step_bipolar():
    pass


def test_get_output_step_unipolar():
    pass


def test_set_output():
    pass


def test_set_neighbours():
    p = Perceptron(ActivationFunctions.IDENTITY)
    p2 = Perceptron(ActivationFunctions.IDENTITY)
    p3 = Perceptron(ActivationFunctions.IDENTITY)
    p.set_neighbours([p2, p3])
    assert p.left_neightbours == [p.inner_neighbour, p2, p3]
    with pytest.raises(ValueError):
        p.set_neighbours([p2, p3, p])
    # not sure why it's protected from setting the same perceptron as neighbour twice
    with pytest.raises(ValueError):
        p.set_neighbours([p2, p2])
    with pytest.raises(ValueError):
        p.set_neighbours([p2])


def test_add_neightbour():
    pass


def test_set_id():
    pass


def test_get_dict():
    pass
