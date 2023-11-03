from typing import Literal
import sys
import os
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)
from src.NetworkEnums import LayerTypes, ActivationFunctions    # noqa: E402
from src.NetworkInput import NetworkInput                       # noqa: E402
from src.NetworkOutput import NetworkOutput                     # noqa: E402
from src.Perceptron import Perceptron                           # noqa: E402


class Layer:
    def __init__(self,
                 layer_type: LayerTypes,
                 left_layer: 'Layer' = None,
                 right_layer: 'Layer' = None) -> None:
        self.id = 'L/?/?/'
        self.children: list = []
        self.layer_type: LayerTypes = layer_type
        self.right_layer: Layer = right_layer
        self.left_layer: Layer = left_layer
        self.__children_functions: list[ActivationFunctions] = []
        self.__layer_num: int = None

    def get_child_count(self) -> int:
        return len(self.children)

    def get_children(self) -> list:
        return self.children

    def get_child(self, index: int) -> 'Layer':
        if index >= len(self.children) or index < 0:
            raise ValueError(
                f"Index out of range. Got {index} | Expected 0 - {len(self.children) - 1}")
        return self.children[index]

    def add_child(self, child) -> None:
        if self.layer_type == LayerTypes.INPUT and not isinstance(child, NetworkInput):
            raise ValueError(
                f"Incorrect child argument. Got {type(child)}. Expected: {NetworkInput}")
        elif self.layer_type == LayerTypes.OUTPUT and not isinstance(child, NetworkOutput):
            raise ValueError(
                f"Incorrect child argument. Got {type(child)}. Expected: {NetworkOutput}")
        elif self.layer_type == LayerTypes.PERC and not isinstance(child, Perceptron):
            raise ValueError(
                f"Incorrect child argument. Got {type(child)}. Expected: {Perceptron}")
        self.children.append(child)
        if isinstance(child, Perceptron):
            self.__children_functions.append(child.activation_function)
        child.set_id(self.__layer_num, len(self.children))
        self.__update_id()

    def add_children(self, children: list) -> None:
        if not isinstance(children, list):
            raise ValueError(
                f"Incorrect children argument. Got {type(children)}. Expected: {list}")
        for child in children:
            self.add_child(child)

    def set_id(self, layer: int) -> None:
        self.__layer_num = layer
        l_type = 'I' if self.layer_type == LayerTypes.INPUT\
            else 'P' if self.layer_type == LayerTypes.PERC else 'O'
        self.id: str = f'L/{l_type}/{layer}/{len(self.children)}'

    def __update_id(self):
        if self.__layer_num is None:
            return
        l_type: Literal['I', 'H', 'O'] = 'I' if self.layer_type == LayerTypes.INPUT\
            else 'H' if self.layer_type == LayerTypes.PERC else 'O'
        self.id: str = f'L/{l_type}/{self.__layer_num}/{len(self.children)}'

    def set_children_ids(self) -> None:
        for i, child in enumerate(self.children):
            child.set_id(self.__layer_num, i + 1)

    def get_sums(self) -> list[float]:
        return [item.calc_sum() for item in self.children]

    def set_children_functions(self, activation_function: ActivationFunctions) -> None:
        self.__children_functions = [
            activation_function for _ in range(len(self.children))]
        for child in self.children:
            child.activation_function = activation_function

    def set_child_weights(self, index: int, weights: list[float]) -> None:
        if index >= len(self.children) or index < 0:
            raise ValueError(
                f"Index out of range. Got {index} | Expected 0 - {len(self.children) - 1}")
        if len(weights) != len(self.left_layer.children) + 1:
            raise ValueError(
                f"Weights count mismatch. Got {len(weights)} | Expected {len(self.left_layer.children)} + 1 (own weight)")
        self.children[index].weights = weights

    def set_children_weights(self, weights: list[list[float]]) -> None:
        if len(weights) != len(self.children):
            raise ValueError(
                f"Weights count mismatch. Got {len(weights)} | Expected {len(self.children)}")
        for i, weight in enumerate(weights):
            self.set_child_weights(i, weight)

    def set_children_functions_by_list(self, activation_functions: list[ActivationFunctions]) -> None:
        if len(activation_functions) != len(self.children):
            raise ValueError(
                f"Activation functions count mismatch. Got {len(activation_functions)} | Expected {len(self.children)}")
        self.__children_functions = [
            ActivationFunctions(i) for i in activation_functions]
        for child, activation_function in zip(self.children, activation_functions):
            child.activation_function = ActivationFunctions(
                activation_function)

    def set_child_function(self, index: int, activation_function: ActivationFunctions) -> None:
        if index >= len(self.children) or index < 0:
            raise ValueError(
                f"Index out of range. Got {index} | Expected 0 - {len(self.children) - 1}")
        self.__children_functions[index] = activation_function
        self.children[index].activation_function = activation_function

    def get_children_functions(self) -> list[ActivationFunctions]:
        return self.__children_functions

    def debug_indepth(self) -> None:
        print(f"Layer {self.id}")
        print(f"Layer type: {self.layer_type}")
        print(f"Left layer: {self.left_layer}")
        print(f"Right layer: {self.right_layer}")
        print(f"Children count: {len(self.children)}")
        print(f"Children: {self.children}")
        print(f"Children functions: {self.__children_functions}")
        print("Children analysis:")
        for child in self.children:
            print('=' * 20)
            print(f"Child {child.id}")
            print(child.activation_function)
            print("Weights: ", child.weights)
            print("Output with activation function: ", child.get_output())
            print("On left side: ", child.left_neightbours)
            print(
                "Note to self: first item on left side is inner value to get inner weight * inner value (Bias)")

    def validate(self) -> None:
        if not isinstance(self.layer_type, LayerTypes):
            raise ValueError(
                f"Incorrect layer_type. Expected {LayerTypes}. Got {type(self.layer_type)}")

        if self.layer_type == LayerTypes.INPUT:
            if self.left_layer is not None:
                raise ValueError(
                    f"Input layer doesn't have left side layer. Got {self.left_layer}")
            if self.right_layer is None:
                raise ValueError(
                    "Input layer can't have empty right side layer")
        elif self.layer_type == LayerTypes.OUTPUT:
            if self.left_layer is None:
                raise ValueError("Output layer needs left side layer.")
            if self.right_layer is not None:
                raise ValueError(
                    f"Output layer doesn't have right side layer. Got {self.right_layer}")
        elif self.layer_type == LayerTypes.PERC:
            if self.left_layer is None:
                raise ValueError("Perceptron layer needs left side layer.")
            if self.right_layer is None:
                raise ValueError("Perceptron layer needs right side layer.")

        if not isinstance(self.left_layer, Layer) and self.left_layer is not None:
            raise ValueError(
                f"Incorrect left layer. Expected {Layer}. Got {type(self.left_layer)}")
        elif not isinstance(self.right_layer, Layer) and self.right_layer is not None:
            raise ValueError(
                f"Incorrect right layer. Expected {Layer}. Got {type(self.right_layer)}")

    def get_dict(self) -> dict:
        obj_dict = {}
        obj_dict['id'] = self.id
        obj_dict['layer-type'] = self.layer_type
        obj_dict['left-layer'] = self.left_layer
        obj_dict['right_layer'] = self.right_layer
        obj_dict['children'] = self.children
        return obj_dict

    def __repr__(self) -> str:
        return self.id
