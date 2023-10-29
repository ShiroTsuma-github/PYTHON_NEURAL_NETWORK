import pytest
import os
import sys
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

from src.NetworkInput import NetworkInput   # noqa: E402


def test_output():
    input = NetworkInput()
    assert input.output == 0


def test_set_id():
    input = NetworkInput()
    input.set_id(1, 2)
    assert input.id == 'I/1/2'


def test_output_setter():
    input = NetworkInput()
    input.output = 5
    assert input.output == 5
    assert input.previous_outputs == [0, 5]

    for i in range(50):
        input.output = i

    assert input.previous_outputs == list(range(0, 50))

    for i in range(50, 100):
        input.output = i

    assert input.previous_outputs == list(range(50, 100))


def test_output_setter_incorrect_data():
    input = NetworkInput()
    with pytest.raises(ValueError):
        input.output = 'string'


def test_repr():
    input = NetworkInput()
    assert repr(input) == 'I/?/?'
    input.set_id(10, 13)
    assert repr(input) == 'I/10/13'
