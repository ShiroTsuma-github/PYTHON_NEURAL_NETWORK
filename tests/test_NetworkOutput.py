import os
import sys
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

from src.NetworkOutput import NetworkOutput   # noqa: E402


def test_output():
    output = NetworkOutput()
    assert output.output == 0

    output.output = 1
    assert output.output == 1

    output.output = 2.5
    assert output.output == 2.5

    try:
        output.output = "invalid"
    except ValueError:
        pass
    else:
        assert False, "Expected ValueError"


def test_set_id():
    output = NetworkOutput()
    output.set_id(1, 2)
    assert output.id == "O/1/2"


def test_previous_outputs():
    output = NetworkOutput()
    assert output.previous_outputs == [0]

    output.output = 1
    assert output.previous_outputs == [0, 1]

    output.output = 2
    assert output.previous_outputs == [0, 1, 2]

    for i in range(50):
        output.output = i

    assert output.output == 49
    assert len(output.previous_outputs) == 50
    assert output.previous_outputs[0] == 0
    assert output.previous_outputs[-1] == 49


def test_repr():
    output = NetworkOutput()
    assert repr(output) == "O/?/?"

    output.set_id(1, 2)
    assert repr(output) == "O/1/2"
