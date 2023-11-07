from typing import Literal
from math import exp, log
import sys
import os
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)
from src.NetworkInput import NetworkInput       # noqa: E402
from src.NetworkEnums import ActivationFunctions    # noqa: E402


class Perceptron:
    def __init__(self,
                 activation_function: int,
                 inner_weight=0,
                 step_bipolar_threshold=0,
                 identity_a=1,
                 parametric_a=0.1) -> None:
        self.__id = 'P/?/?'
        self.activation_function = activation_function
        self.__inner_weight = inner_weight
        self.__weights: list = [self.__inner_weight]
        self.previous_weights: list[list[float]] = []
        self.__output: float = 0
        self.previous_outputs: list[float] = []
        self.inner_neighbour = NetworkInput(1)
        self.__step_bipolar_threshold = step_bipolar_threshold
        self.__identity_a = identity_a
        self.__parametric_a = parametric_a
        self.left_neightbours: 'list[Perceptron]' = [self.inner_neighbour]
        self.validate()

    @property
    def id(self):
        return self.__id

    @id.setter
    def id(self, value):
        raise Exception("Cannot set id manually")

    @property
    def output(self):
        return self.__output

    @output.setter
    def output(self, value):
        if not isinstance(value, (int, float)):
            raise ValueError(f"Incorrect output. {self.id} | {value}")
        self.__output = value
        self.previous_outputs.append(value)
        if len(self.previous_outputs) > 50:
            self.previous_outputs.pop(0)

    @property
    def weights(self):
        return self.__weights

    @weights.setter
    def weights(self, value):
        if not isinstance(value, (list)):
            raise ValueError(f"Incorrect weights. {self.id} | {value}")
        if len(value) != len(self.__weights):
            raise ValueError("Mismatch between size of weights."
                             f"Prev: {len(self.__weights)} | New: {len(value)}")
        if not all(isinstance(item, (int, float)) for item in value):
            raise ValueError(f"Incorrect weights. {self.id} | {value}")
        self.__weights = value
        self.__inner_weight = value[0]
        self.previous_weights.append(value)
        if len(self.previous_weights) > 50:
            self.previous_weights.pop(0)

    def __add_weight(self, value):
        if not isinstance(value, (int, float)):
            raise ValueError(f"Incorrect weight. {self.id} | {value}")
        self.__weights.append(value)
        self.previous_weights = [self.weights]

    def modify_weights(self, expected_output, training_set):
        new = []
        training_set = [1] + training_set
        for i, item in enumerate(self.__weights):
            new.append(item + training_set[i] * expected_output)
        self.weights = new

    def validate(self, explicit=False):
        if len(self.weights) != len(self.left_neightbours):
            raise Exception(f'Perceptron id: {self.id}')
        if explicit:
            print(
                f"Weight count and Neigbour count match {len(self.left_neightbours)} | {len(self.weights) - 1}")

    def add_neightbour(self, neightbour, weight: float = 0):
        if not isinstance(neightbour, (Perceptron, NetworkInput)):
            raise ValueError("Neighbour is not instance "
                             "of Perceptron or Network Input. "
                             f"Instead {type(neightbour)}")
        if neightbour in self.left_neightbours:
            raise ValueError("Neighbour already exists")
        if neightbour == self:
            raise ValueError("Cannot add self as neighbour")
        self.left_neightbours.append(neightbour)
        self.__add_weight(weight)

    def set_neighbours(self, neighbours):
        self.left_neightbours = [self.inner_neighbour]
        self.__weights = [self.__inner_weight]
        self.previous_weights = []
        for neighbour in neighbours:
            self.add_neightbour(neighbour)

    def calc_sum(self) -> float:
        sum = 0
        for perc, weight in zip(self.left_neightbours, self.weights):
            sum += perc.output * weight
        return round(sum, 10)

    def set_id(self, layer: int, position: int) -> None:
        self.__id: str = f'P/{layer}/{position}'
        self.inner_neighbour.set_id(layer, position)

    def calc_step_unipolar(self) -> Literal[1, 0]:
        if self.calc_sum() > 0:
            return 1
        return 0

    def calc_step_unipolar_der(self):
        return 0

    def calc_step_bipolar(self) -> Literal[1, -1]:
        if self.calc_sum() >= self.__step_bipolar_threshold:
            return 1
        return -1

    def calc_step_bipolar_der(self):
        return 0

    def calc_identity(self) -> float:
        return self.__identity_a * self.calc_sum()

    def calc_identity_der(self):
        return self.__identity_a

    def calc_sigmoid_unipolar(self) -> float:
        return 1 / (1 + exp(-self.calc_sum()))

    def calc_sigmoid_unipolar_der(self) -> float:
        return self.calc_sigmoid_unipolar() * (1 - self.calc_sigmoid_unipolar())

    def calc_sigmoid_bipolar(self) -> float:
        return -1 + 2 / (1 + exp(self.calc_sum()))

    def calc_sigmoid_bipolar_der(self) -> float:
        res = self.calc_sigmoid_bipolar()
        return 0.5 * (1 + res) * (1 - res)

    def calc_relu(self) -> float:
        return max(0, self.calc_sum())

    def calc_relu_der(self) -> float:
        if self.calc_sum() >= 0:
            return 1
        return 0

    def calc_relu_leaky(self) -> float:
        sum_ = self.calc_sum()
        return max(0.01 * sum_, sum_)

    def calc_relu_leaky_der(self) -> float:
        sum_ = self.calc_sum()
        if sum_ >= 0:
            return 1
        return 0.01

    def calc_relu_parametric(self) -> float:
        sum_ = self.calc_sum()
        if sum_ > 0:
            return sum_
        return self.__parametric_a * sum_

    def calc_relu_parametric_der(self) -> float:
        sum_ = self.calc_sum()
        if sum_ >= 0:
            return 1
        return self.__parametric_a

    def calc_softplus(self) -> float:
        return log(1 + exp(self.calc_sum()))

    def calc_softplus_der(self) -> float:
        return 1 / (1 + exp(-self.calc_sum()))

    def get_output(self) -> float | Literal[1, -1, 0]:
        if self.activation_function == ActivationFunctions.IDENTITY:
            return self.calc_identity()
        elif self.activation_function == ActivationFunctions.RELU:
            return self.calc_relu()
        elif self.activation_function == ActivationFunctions.SIGMOID_BIPOLAR:
            return self.calc_sigmoid_bipolar()
        elif self.activation_function == ActivationFunctions.SIGMOID_UNIPOLAR:
            return self.calc_sigmoid_unipolar()
        elif self.activation_function == ActivationFunctions.SOFTPLUS:
            return self.calc_softplus()
        elif self.activation_function == ActivationFunctions.STEP_BIPOLAR:
            return self.calc_step_bipolar()
        elif self.activation_function == ActivationFunctions.STEP_UNIPOLAR:
            return self.calc_step_unipolar()
        elif self.activation_function == ActivationFunctions.RELU_LEAKY:
            return self.calc_relu_leaky()
        elif self.activation_function == ActivationFunctions.RELU_PARAMETRIC:
            return self.calc_relu_parametric()
        else:
            raise ValueError("Could not match activation function")

    def get_output_der(self) -> float | Literal[1, -1, 0]:
        if self.activation_function == ActivationFunctions.IDENTITY:
            return self.calc_identity_der()
        elif self.activation_function == ActivationFunctions.RELU:
            return self.calc_relu_der()
        elif self.activation_function == ActivationFunctions.SIGMOID_BIPOLAR:
            return self.calc_sigmoid_bipolar_der()
        elif self.activation_function == ActivationFunctions.SIGMOID_UNIPOLAR:
            return self.calc_sigmoid_unipolar_der()
        elif self.activation_function == ActivationFunctions.SOFTPLUS:
            return self.calc_softplus_der()
        elif self.activation_function == ActivationFunctions.STEP_BIPOLAR:
            return self.calc_step_bipolar_der()
        elif self.activation_function == ActivationFunctions.STEP_UNIPOLAR:
            return self.calc_step_unipolar_der()
        elif self.activation_function == ActivationFunctions.RELU_LEAKY:
            return self.calc_relu_leaky_der()
        elif self.activation_function == ActivationFunctions.RELU_PARAMETRIC:
            return self.calc_relu_parametric_der()
        else:
            raise ValueError("Could not match activation function")

    def get_set_output(self):
        self.output = self.get_output()
        return self.output

    # def get_set_output_der(self):

    def set_output(self, value):
        self.output = value

    def get_dict(self) -> dict:
        obj_dict = {}
        obj_dict['id'] = self.id
        obj_dict['output'] = self.output
        obj_dict['activation-function'] = self.activation_function
        obj_dict['weights'] = self.__weights
        obj_dict['inner-weight'] = self.__inner_weight
        obj_dict['left-side-U'] = [item.output for item in self.left_neightbours[1:]]
        obj_dict['inner-U'] = self.inner_neighbour.output
        obj_dict['U-id'] = [item.id for item in self.left_neightbours]
        return obj_dict

    def __repr__(self) -> str:
        return self.__id
