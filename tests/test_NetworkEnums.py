import os
import sys
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)

from src.NetworkEnums import LayerTypes, ActivationFunctions   # noqa: E402


def test_LayerTypes():
    assert len(LayerTypes) == 3
    assert LayerTypes.INPUT.value == 1
    assert LayerTypes.OUTPUT.value == 2
    assert LayerTypes.PERC.value == 3


def test_ActivationFunctions():
    assert len(ActivationFunctions) == 9
    assert ActivationFunctions.STEP_UNIPOLAR.value == 1
    assert ActivationFunctions.STEP_BIPOLAR.value == 2
    assert ActivationFunctions.SIGMOID_UNIPOLAR.value == 3
    assert ActivationFunctions.SIGMOID_BIPOLAR.value == 4
    assert ActivationFunctions.IDENTITY.value == 5
    assert ActivationFunctions.RELU.value == 6
    assert ActivationFunctions.RELU_LEAKY.value == 7
    assert ActivationFunctions.RELU_PARAMETRIC.value == 8
    assert ActivationFunctions.SOFTPLUS.value == 9
