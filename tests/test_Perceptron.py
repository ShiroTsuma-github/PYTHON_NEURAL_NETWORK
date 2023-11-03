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
    with pytest.raises(ValueError):
        p.weights = [1, "a"]
    # do ogarnięcia

    for i in range(100):
        p.weights = [i, i + 1]

    assert p.weights == [99, 100]
    assert len(p.previous_weights) == 50
    assert p.previous_weights[0] == [50, 51]


def test_add_neighbour():
    p = Perceptron(ActivationFunctions.IDENTITY)
    p2 = Perceptron(ActivationFunctions.IDENTITY)
    p3 = Perceptron(ActivationFunctions.IDENTITY)
    p.add_neightbour(p2)
    assert p.left_neightbours == [p.inner_neighbour, p2]
    p.add_neightbour(p3)
    assert p.left_neightbours == [p.inner_neighbour, p2, p3]
    with pytest.raises(ValueError):
        p.add_neightbour(p)
    with pytest.raises(ValueError):
        p.add_neightbour(p2)
    with pytest.raises(ValueError):
        p.add_neightbour(p3)
    with pytest.raises(ValueError):
        p.add_neightbour(1)
    with pytest.raises(ValueError):
        p.add_neightbour("a")
    with pytest.raises(ValueError):
        p.add_neightbour([1, 2, 3])


def test_get_output_identity():
    p1 = Perceptron(ActivationFunctions.IDENTITY)
    p2 = Perceptron(ActivationFunctions.IDENTITY)
    p3 = Perceptron(ActivationFunctions.IDENTITY)
    p1.set_neighbours([p2, p3])
    p1.weights = [1, 2, 3]
    p2.output = 2
    p3.output = 3
    # 1 + 2*2 + 3*3 = 14
    # identity(14) = 14
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
    # relu(8) = 8
    assert p1.get_output() == 8
    p2.output = 1
    p3.output = 0.6
    p1.weights = [-0.1, -0.7, 0.4]
    # -0.1 + -0.7*1 + 0.4*0.6 = -0.38
    # relu(-0.38) = 0
    assert p1.get_output() == 0
    assert p1.get_output() == p1.calc_relu()


def test_get_output_relu_leaky():
    p1 = Perceptron(ActivationFunctions.RELU_LEAKY)
    p2 = Perceptron(ActivationFunctions.RELU_LEAKY)
    p3 = Perceptron(ActivationFunctions.RELU_LEAKY)
    p1.set_neighbours([p2, p3])
    p1.weights = [1, 2, 3]
    p2.output = -1
    p3.output = 3
    # 1 + 2*-1 + 3*3 = 8
    # relu_leaky(8) = 8
    assert p1.get_output() == 8
    p2.output = 1
    p3.output = 0.6
    p1.weights = [-0.1, -0.7, 0.4]
    # -0.1 + -0.7*1 + 0.4*0.6 = -0.056
    # relu_leaky(-0.38) = -0.038
    assert p1.get_output() == -0.05600000000000001
    assert p1.get_output() == p1.calc_relu_leaky()
    p2.output = -1
    p3.output = -3
    p1.weights = [1, 2, 3]
    # 1 - 2 - 9 = -10
    # relu_leaky(-10) = -1 ( 0.1 * -10 = -1) (NISKIE A)
    assert p1.get_output() == -1
    assert p1.get_output() == p1.calc_relu_leaky()


def test_get_output_relu_parametric():
    p1 = Perceptron(ActivationFunctions.RELU_PARAMETRIC)
    p2 = Perceptron(ActivationFunctions.RELU_PARAMETRIC)
    p3 = Perceptron(ActivationFunctions.RELU_PARAMETRIC)
    p1.set_neighbours([p2, p3])
    p1.weights = [1, 2, 3]
    p2.output = -1
    p3.output = 3
    # 1 + 2*-1 + 3*3 = 8
    # relu_parametric(8) = 8
    assert p1.get_output() == 8
    p2.output = 1
    p3.output = 0.6
    p1.weights = [-0.1, -0.7, 0.4]
    # -0.1 + -0.7*1 + 0.4*0.6 = -0.38
    # relu_parametric(-0.38) = -0.038
    assert p1.get_output() == -0.05600000000000001
    assert p1.get_output() == p1.calc_relu_parametric()
    p2.output = -1
    p3.output = -3
    p1.weights = [1, 2, 3]
    # 1 - 2 - 9 = -10
    # relu_parametric(-10) = -10
    assert p1.get_output() == -1
    assert p1.get_output() == p1.calc_relu_parametric()

    p1 = Perceptron(ActivationFunctions.RELU_PARAMETRIC, parametric_a=0.02)
    p1.set_neighbours([p2, p3])
    p1.weights = [1, 2, 3]
    p2.output = -12
    p3.output = -3
    # 1 + 2*-12 + 3*-3 = -32
    assert p1.get_output() == -0.64


def test_get_output_sigmoid_bipolar():
    p1 = Perceptron(ActivationFunctions.SIGMOID_BIPOLAR)
    p2 = Perceptron(ActivationFunctions.SIGMOID_BIPOLAR)
    p3 = Perceptron(ActivationFunctions.SIGMOID_BIPOLAR)
    p1.set_neighbours([p2, p3])
    p1.weights = [2, 2, 3]
    p2.output = -1
    p3.output = 3
    # 1 + 2*-1 + 3*3 = 8
    # sigmoid_bipolar(8) = -0.9997532108480275
    assert p1.get_output() == -0.9997532108480275
    p2.output = 1
    p3.output = 1
    p1.weights = [-0.1, -0.7, 0.4]
    assert p1.get_output() == 0.197375320224904
    assert p1.get_output() == p1.calc_sigmoid_bipolar()


def test_get_output_sigmoid_unipolar():
    p = Perceptron(ActivationFunctions.SIGMOID_UNIPOLAR)
    p2 = Perceptron(ActivationFunctions.SIGMOID_UNIPOLAR)
    p3 = Perceptron(ActivationFunctions.SIGMOID_UNIPOLAR)
    p.set_neighbours([p2, p3])
    p.weights = [1, 2, 3]
    p2.output = 2
    p3.output = 3
    # 1 + 2*2 + 3*3 = 14
    # sigmoid_unipolar(14) = 0.9999991684719722
    assert p.get_output() == 0.9999991684719722
    assert p.get_output() == p.calc_sigmoid_unipolar()
    p2.output = 1
    p3.output = 0.6
    p.weights = [-0.1, -0.7, 0.4]
    # -0.1 + -0.7*1 + 0.4*0.6 = -0.38
    # sigmoid_unipolar(-0.38) = 0.36354745971843366
    assert p.get_output() == 0.36354745971843366
    assert p.get_output() == p.calc_sigmoid_unipolar()


def test_get_output_softplus():
    p = Perceptron(ActivationFunctions.SOFTPLUS)
    p2 = Perceptron(ActivationFunctions.SOFTPLUS)
    p3 = Perceptron(ActivationFunctions.SOFTPLUS)
    p.set_neighbours([p2, p3])
    p.weights = [1, 2, 3]
    p2.output = 2
    p3.output = 3
    # 1 + 2*2 + 3*3 = 14
    # softplus(14) = 14.000000831528373
    assert p.get_output() == 14.000000831528373
    assert p.get_output() == p.calc_softplus()
    p2.output = 1
    p3.output = 0.6
    p.weights = [-0.1, -0.7, 0.4]
    # -0.1 + -0.7*1 + 0.4*0.6 = -0.38
    # softplus(-0.38) = 0.4518454273443063
    assert p.get_output() == 0.4518454273443063
    assert p.get_output() == p.calc_softplus()


def test_get_output_step_bipolar():
    p = Perceptron(ActivationFunctions.STEP_BIPOLAR)
    p2 = Perceptron(ActivationFunctions.STEP_BIPOLAR)
    p3 = Perceptron(ActivationFunctions.STEP_BIPOLAR)
    p.set_neighbours([p2, p3])
    p.weights = [1, 2, 3]
    p2.output = 2
    p3.output = 3
    # 1 + 2*2 + 3*3 = 14
    # step_bipolar(14) = 1
    assert p.get_output() == 1
    assert p.get_output() == p.calc_step_bipolar()
    p2.output = 1
    p3.output = 0.6
    p.weights = [-0.1, -0.7, 0.4]
    # -0.1 + -0.7*1 + 0.4*0.6 = -0.38
    # step_bipolar(-0.38) = -1
    assert p.get_output() == -1


def test_get_output_step_unipolar():
    p = Perceptron(ActivationFunctions.STEP_UNIPOLAR)
    p2 = Perceptron(ActivationFunctions.STEP_UNIPOLAR)
    p3 = Perceptron(ActivationFunctions.STEP_UNIPOLAR)
    p.set_neighbours([p2, p3])
    p.weights = [1, 2, 3]
    p2.output = 2
    p3.output = 3
    # 1 + 2*2 + 3*3 = 14
    # step_unipolar(14) = 1
    assert p.get_output() == 1
    assert p.get_output() == p.calc_step_unipolar()
    p2.output = 1
    p3.output = 0.6
    p.weights = [-0.1, -0.7, 0.4]
    # -0.1 + -0.7*1 + 0.4*0.6 = -0.38
    # step_unipolar(-0.38) = 0
    assert p.get_output() == 0
    assert p.get_output() == p.calc_step_unipolar()


def test_set_output():
    p = Perceptron(ActivationFunctions.IDENTITY)
    with pytest.raises(ValueError):
        p.set_output("a")
    with pytest.raises(ValueError):
        p.set_output([1, 2, 3])
    p.set_output(0.3)
    assert p.output == 0.3
    p.set_output(1)
    assert p.output == 1
    p.set_output(0)
    assert p.previous_outputs == [0.3, 1, 0]


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


def test_set_id():
    p = Perceptron(ActivationFunctions.IDENTITY)
    with pytest.raises(Exception):
        p.id = "a"
    with pytest.raises(Exception):
        p.id = [1, 2, 3]
    with pytest.raises(TypeError):
        p.set_id(-1)
    p.set_id(1, 1)
    assert p.id == 'P/1/1'
    p.set_id(3, 8)
    assert p.id == 'P/3/8'


def test_incorrect_activation_function():
    p = Perceptron('abc')
    with pytest.raises(ValueError):
        p.get_output()
    p = Perceptron([1, 2, 3])
    with pytest.raises(ValueError):
        p.get_output()


def test_get_dict():
    p = Perceptron(ActivationFunctions.IDENTITY)
    p2 = Perceptron(ActivationFunctions.IDENTITY)
    p3 = Perceptron(ActivationFunctions.IDENTITY)
    p.set_neighbours([p2, p3])
    p.set_id(0, 0)
    p2.set_id(1, 1)
    p3.set_id(2, 2)
    p.weights = [1, 2, 3]
    p2.output = 2
    p3.output = 3
    # 1 + 2*2 + 3*3 = 14
    # identity(14) = 14
    assert p.get_dict() == {
        'id': 'P/0/0',
        'output': 0,
        'activation-function': ActivationFunctions.IDENTITY,
        'weights': [1, 2, 3],
        'inner-weight': 1,
        'left-side-U': [2, 3],
        'inner-U': 1,
        'U-id': ['I/0/0', 'P/1/1', 'P/2/2']
    }
    p2.output = 1
    p3.output = 0.6
    p.weights = [-0.1, -0.7, 0.4]
    # -0.1 + -0.7*1 + 0.4*0.6 = -0.38
    # relu(-0.38) = 0
    p.set_output(p.get_output())
    assert p.get_dict() == {
        'id': 'P/0/0',
        'output': -0.56,
        'activation-function': ActivationFunctions.IDENTITY,
        'weights': [-0.1, -0.7, 0.4],
        'inner-weight': -0.1,
        'left-side-U': [1, 0.6],
        'inner-U': 1,
        'U-id': ['I/0/0', 'P/1/1', 'P/2/2']
    }


def test_repr():
    p = Perceptron(ActivationFunctions.IDENTITY)
    assert repr(p) == "P/?/?"
    p.set_id(0, 0)
    assert repr(p) == "P/0/0"
    p.set_id(1, 1)
    assert repr(p) == "P/1/1"
