from typing import Literal
from math import exp, log
import sys
import os
from random import randint
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)
from src.NetworkInput import NetworkInput       # noqa: E402
from src.NetworkEnums import ActivationFunctions    # noqa: E402


class Perceptron:
    def __init__(self,
                 activation_function: int,
                 inner_weight=0,
                 learning_rate=0.1,
                 momentum=0,
                 step_bipolar_threshold=0,
                 identity_a=1,
                 parametric_a=0.1) -> None:
        self.__id = 'P/?/?'
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.activation_function = activation_function
        self.__inner_weight = inner_weight
        self.__weights: list = [self.__inner_weight]
        self.previous_weights: list[list[float]] = []
        self.__output: float = 0
        self.error: float = 0
        self.previous_difference: float = []
        self.previous_outputs: list[float] = []
        self.inner_neighbour = NetworkInput(1)
        self.__velocity_weight = []
        self.__step_bipolar_threshold = step_bipolar_threshold
        self.__identity_a = identity_a
        self.__parametric_a = parametric_a
        self.left_neighbours: 'list[Perceptron]' = [self.inner_neighbour]
        self.right_neighbours: 'list[Perceptron]' = []
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
        self.previous_weights.append(value.copy())
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

    def update_weights(self):
        new = []
        if len(self.__velocity_weight) == 0:
            self.__velocity_weight = [0] * len(self.__weights)
        for i, item in enumerate(self.__weights):
            self.__velocity_weight[i] = self.momentum * self.__velocity_weight[i] + self.learning_rate * self.error * self.left_neighbours[i].output
            new.append(item + self.__velocity_weight[i])
        self.weights = new

    def randomize_weights_around_1(self):
        new = []
        for item in self.__weights:
            new.append(randint(-100, 100) / 100)
        self.weights = new

    def randomize_weights_around_10(self):
        new = []
        for item in self.__weights:
            new.append(randint(-1000, 1000) / 100)
        self.weights = new

    def get_previous_difference(self, index):
        if len(self.previous_weights) < 2:
            return 0
        return self.previous_weights[-1][index] - self.previous_weights[-2][index]

    def get_max_weight_change(self):
        if len(self.previous_weights) < 2:
            return 0
        return max([abs(item - self.previous_weights[-2][i]) for i, item in enumerate(self.previous_weights[-1])])

    def calc_error(self, index, expected_output=None):
        if expected_output is None:
            self.error = sum([perc.error * perc.weights[index] for perc in self.right_neighbours]) * self.get_output_der()
        else:
            self.error = (expected_output - self.output) * self.get_output_der()
        return self.error

    def validate(self, explicit=False):
        if len(self.weights) != len(self.left_neighbours):
            raise Exception(f'Perceptron id: {self.id}')
        if explicit:
            print(
                f"Weight count and Neigbour count match {len(self.left_neighbours)} | {len(self.weights) - 1}")

    def add_neighbour(self, neighbour, weight: float = 0):
        if not isinstance(neighbour, (Perceptron, NetworkInput)):
            raise ValueError("Neighbour is not instance "
                             "of Perceptron or Network Input. "
                             f"Instead {type(neighbour)}")
        if neighbour in self.left_neighbours:
            raise ValueError("Neighbour already exists")
        if neighbour == self:
            raise ValueError("Cannot add self as neighbour")
        self.left_neighbours.append(neighbour)
        self.__add_weight(weight)

    def set_neighbours(self, neighbours):
        self.left_neighbours = [self.inner_neighbour]
        self.__weights = [self.__inner_weight]
        self.previous_weights = []
        for neighbour in neighbours:
            self.add_neighbour(neighbour)

    def set_right_neighbours(self, neighbours):
        self.right_neighbours = neighbours

    def calc_sum(self) -> float:
        sum_ = 0
        for perc, weight in zip(self.left_neighbours, self.__weights):
            sum_ += perc.output * weight
        return sum_

    def set_id(self, layer: int, position: int) -> None:
        self.__id: str = f'P/{layer}/{position}'
        self.inner_neighbour.set_id(layer, position)

    def calc_step_unipolar(self) -> Literal[1, 0]:
        if self.calc_sum() > 0:
            return 1
        return 0

    def calc_step_unipolar_der(self):
        raise TypeError("Step unipolar derivative is not defined")

    def calc_step_bipolar(self) -> Literal[1, -1]:
        if self.calc_sum() >= self.__step_bipolar_threshold:
            return 1
        return -1

    def calc_step_bipolar_der(self):
        raise TypeError("Step bipolar derivative is not defined")

    def calc_identity(self) -> float:
        return self.__identity_a * self.calc_sum()

    def calc_identity_der(self):
        return self.__identity_a

    def calc_sigmoid_unipolar(self) -> float:
        return 1 / (1 + exp(-self.calc_sum()))

    def calc_sigmoid_unipolar_der(self) -> float:
        res = self.calc_sigmoid_unipolar()
        return res * (1 - res)

    def calc_sigmoid_bipolar(self) -> float:
        return -1 + 2 / (1 + exp(-self.calc_sum()))

    def calc_sigmoid_bipolar_der(self) -> float:
        res = self.calc_sigmoid_bipolar()
        return 0.5 * (1 - res * res)

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

    def get_output(self):
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

    def get_output_der(self):
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
        obj_dict['left-side-U'] = [item.output for item in self.left_neighbours[1:]]
        obj_dict['inner-U'] = self.inner_neighbour.output
        obj_dict['U-id'] = [item.id for item in self.left_neighbours]
        return obj_dict

    def __repr__(self) -> str:
        return self.__id
