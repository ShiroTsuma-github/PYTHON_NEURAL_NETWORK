from enum import Enum, auto


class LayerTypes(Enum):
    INPUT = auto()
    OUTPUT = auto()
    PERC = auto()


class ActivationFunctions(Enum):
    STEP_UNIPOLAR = auto()
    STEP_BIPOLAR = auto()
    SIGMOID_UNIPOLAR = auto()
    SIGMOID_BIPOLAR = auto()
    IDENTITY = auto()
    RELU = auto()
    RELU_LEAKY = auto()
    RELU_PARAMETRIC = auto()
    SOFTPLUS = auto()