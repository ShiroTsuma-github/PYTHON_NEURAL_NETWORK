import json
from cryptography.fernet import Fernet
import pprint
import csv
import sys
import os
import time
from random import shuffle
ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(ROOT_DIR)
from src.Layer import Layer                                     # noqa: E402
from src.NetworkEnums import LayerTypes, ActivationFunctions    # noqa: E402
from src.NetworkInput import NetworkInput                       # noqa: E402
from src.NetworkOutput import NetworkOutput                     # noqa: E402
from src.Perceptron import Perceptron                           # noqa: E402


class NeuralNetwork:
    def __init__(self, learning_rate=0.1, momentum=0) -> None:
        self.id = 'N/I/?/H/?/P/?/O/?'
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.layer_count: int = 0
        self.layers: list[Layer] = []
        self.__perc_layers: list[Layer] = []
        self.__input_layer: Layer = None
        self.__output_layer: Layer = None
        self.perceptrons_per_layer: list[int] = []
        self.input_count: int = 0
        self.output_count: int = 0
        self.__mov_layer_i: int = 1

    def __update_id(self):
        self.id = f'N/I/{self.__input_layer.get_child_count()}/H/{len(self.__perc_layers)}/P/{self.get_perceptron_count()}/O/{self.__output_layer.get_child_count()}'

    def get_perceptron_count(self) -> int:
        return sum(self.perceptrons_per_layer)

    def get_input_values(self) -> list[float]:
        return [item.output for item in self.__input_layer.get_children()]

    def set_input_values(self, values: list[float]) -> None:
        if len(values) != self.__input_layer.get_child_count():
            raise ValueError(
                f"Input values mismatch. Got {len(values)} | Expected {self.__input_layer.get_child_count()}")
        for value, input in zip(values, self.__input_layer.get_children()):
            input.output = value

    def get_output_values(self) -> list[float]:
        return [item.output for item in self.__output_layer.get_children()]

    def validate_network(self) -> None:
        for layer in self.layers:
            layer.validate()

    def calc_softmax(self) -> list[float]:
        self.__output_layer.get_sums()
        # implement softmax, linear and sigmoid as output functions

    def get_output(self) -> list[float]:
        pass

    def randomize_weights(self, around_ten=False) -> None:
        for layer in self.__perc_layers:
            if around_ten:
                layer.randomize_weights(True)
            else:
                layer.randomize_weights()

    def __load_csv(self, path: str) -> tuple[list[list[float]], list[float]]:
        data = []
        output = []
        with open(path, newline='') as csvfile:
            reader = csv.reader(csvfile, delimiter=',')
            for row in reader:
                if len(row) == 0:
                    continue
                data.append([float(val) for val in row[:-1]])
                output.append(float(row[-1]))
        return (data, output)

    def test(self, inputs: list[float]) -> None:
        self.forward(inputs)
        for item in self.__output_layer.get_children():
            print(item.output)

    def z_score(self, data: list[list[float]]) -> list[list[float]]:
        return NotImplementedError

    def train_single_perceptron(self, csv_path):
        data, output = self.__load_csv(csv_path)
        perceptron = self.__perc_layers[0].get_children()[0]
        layer = self.__perc_layers[0]
        iter_count = 0
        max_iter = len(data) * 100
        newdata = data.copy()
        newoutput = output.copy()
        while True:
            incorrect_data = newdata.copy()
            incorrect_output = newoutput.copy()
            for inputs, expected_output in zip(incorrect_data, incorrect_output):
                self.set_input_values(inputs)
                layer.results = [perceptron.get_set_output()]
                if perceptron.output != expected_output:
                    newdata = data.copy()
                    newoutput = output.copy()
                    perceptron.modify_weights(expected_output, inputs)
                else:
                    newdata.pop(0)
                    newoutput.pop(0)
            if incorrect_data == []:
                break
            elif iter_count > max_iter:
                print('Max iterations reached')
                break
            iter_count += 1

    def train_backpropagation(self, csv_path,
                              limit_iter=100,
                              limit_time_sec=10,
                              error_threshold=0.00001):

        def should_run(end_time, iter_count):
            if limit_iter is None:
                if time.time() < end_time:
                    return True
                return False
            elif limit_time_sec is None:
                if iter_count < limit_iter:
                    return True
                return False
            else:
                if iter_count < limit_iter and time.time() < end_time:
                    return True
                return False

        self.prepare_for_backpropagation()
        data, output = self.__load_csv(csv_path)
        end_time = time.time() + 0 if limit_time_sec is None else limit_time_sec
        iter_count = 0
        while (should_run(end_time, iter_count)):
            iter_count += 1
            indexes = [i for i in range(len(data))]
            shuffle(indexes)
            data = [data[i] for i in indexes]
            output = [output[i] for i in indexes]
            total_error = 0
            for single_data, single_output in zip(data, output):
                self.forward(single_data)
                error = self.propagate_error(single_output)
                total_error += error * error
                self.update_weights()
            average_error = total_error / len(data)
            if iter_count % 100 == 0:
                print(f'Iteration {iter_count} | Average error: {average_error}')
            if average_error < error_threshold:
                print("Error under threshold. Ending at ", iter_count)
                break

    def forward(self, single_data):
        self.set_input_values(single_data)
        for layer in self.__perc_layers:
            layer.calc_outputs()
        self.__perc_layers[-1].forward_output()

    def propagate_error(self, single_output):
        total_error = 0
        total_error += self.__perc_layers[-1].calc_errors(single_output)
        for layer in self.__perc_layers[::-1][1:]:
            total_error += layer.calc_errors()
        return total_error

    def update_weights(self):
        for layer in self.__perc_layers:
            layer.update_children_weights()

    def get_perc_layers(self) -> list[Layer]:
        return self.__perc_layers

    def setup(self, inputs: int = 1, perc_layers: int = 1):
        layer = Layer(LayerTypes.INPUT)
        layer.set_id(self.__mov_layer_i)
        self.__mov_layer_i += 1
        self.layers.append(layer)
        for i in range(perc_layers):
            layer = Layer(LayerTypes.PERC)
            layer.set_id(self.__mov_layer_i)
            self.__mov_layer_i += 1
            self.layers.append(layer)
            self.__perc_layers.append(layer)
        layer = Layer(LayerTypes.OUTPUT)
        layer.set_id(self.__mov_layer_i)
        self.__mov_layer_i += 1
        self.layers.append(layer)
        self.__input_layer = self.layers[0]
        self.__output_layer = self.layers[len(self.layers) - 1]
        for i in range(inputs):
            self.layers[0].add_child(NetworkInput())
        for index in range(len(self.layers)):
            if index == 0:
                self.layers[index].right_layer = self.layers[index + 1]
            elif index == len(self.layers) - 1:
                self.layers[index].left_layer = self.layers[index - 1]
            else:
                self.layers[index].left_layer = self.layers[index - 1]
                self.layers[index].right_layer = self.layers[index + 1]
        self.validate_network()
        self.__update_id()

    def set_perceptrons_per_layer(self, perceptrons_per_layer: list[int]) -> None:
        if len(perceptrons_per_layer) != len(self.__perc_layers):
            raise ValueError(
                f"Got perceptrons for incorrect number of layers. Got {len(perceptrons_per_layer)} | Expected {len(self.__perc_layers)} \
Remember that output count is equal to the number of perceptrons in the last perceptron layer")
        self.perceptrons_per_layer = perceptrons_per_layer
        for layer, perceptron_count in zip(self.__perc_layers, perceptrons_per_layer):
            for i in range(perceptron_count):
                layer.add_child(Perceptron(ActivationFunctions.STEP_UNIPOLAR, learning_rate=self.learning_rate, momentum=self.momentum))
            layer.set_children_ids()
        for _ in range(perceptrons_per_layer[-1]):
            self.__output_layer.add_child(NetworkOutput())

        for layer in self.layers:
            for child in layer.get_children():
                if isinstance(child, Perceptron):
                    child.set_neighbours(layer.left_layer.get_children())
        self.__update_id()

    def prepare_for_backpropagation(self) -> None:
        for layer in self.__perc_layers:
            layer.prepare_for_backpropagation()

    def set_layer_activation_function(self, id, activation_function: ActivationFunctions) -> None:
        layer = self.get_layer_by_index(id)
        layer.set_children_functions(activation_function)

    def get_layer_by_index(self, index: int) -> Layer:
        # check if index is in range
        if index >= 1 and index < len(self.layers) - 1:
            return self.layers[index]
        raise ValueError(
            f"Index out of range. Got {index} | Expected 1 - {len(self.layers) - 2}")

    def get_dict(self, display=True) -> dict:
        obj_dict = {}
        obj_dict['id'] = self.id
        obj_dict['input-layer'] = self.__input_layer
        obj_dict['input-count'] = self.__input_layer.get_child_count()
        obj_dict['input_values'] = self.get_input_values()
        obj_dict['hidden-layers'] = self.__perc_layers
        obj_dict['perceptrons-per-layer'] = self.perceptrons_per_layer
        obj_dict['output-layer'] = self.__output_layer
        obj_dict['output-count'] = self.__output_layer.get_child_count()
        obj_dict['output-values'] = self.get_output_values()
        if display:
            pprint.pprint(obj_dict)
        return obj_dict

    def save_network(self, path):
        perc_dict = {}
        for i, layer in enumerate(self.__perc_layers):
            perc_dict[f'{i + 1}'] = {
                'perceptons_activation_functions': [item.value for item in layer.get_children_functions()],
                'perceptons_weights': [item.weights for item in layer.get_children()]}

        save_dict = {
            'network': {
                'id': self.id,
                'input-count': self.__input_layer.get_child_count(),
                'hidden-layer_count': len(self.__perc_layers),
                'perceptrons-per-layer': self.perceptrons_per_layer,
                'output-count': self.__output_layer.get_child_count(),
                'input-values': self.get_input_values(),
                'output-values': self.get_output_values(),
                'perceptrons_data': perc_dict
            }
        }
        json_string = json.dumps(save_dict)

        cipher_suite = Fernet('iy2hgPRYISdrCh11bmHUPa5yQKCHy7EmR4dgzhh1unE=')
        json_string = cipher_suite.encrypt(json_string.encode('utf-8'))
        with open(path, 'wb') as file:
            file.write(json_string)

    def load_network(self, path):
        with open(path, 'rb') as file:
            data_bytes = file.read()

        cipher_suite = Fernet('iy2hgPRYISdrCh11bmHUPa5yQKCHy7EmR4dgzhh1unE=')
        data_bytes = cipher_suite.decrypt(data_bytes)

        self.__init__()
        json_string = data_bytes.decode('utf-8')
        data = json.loads(json_string)
        network_data = data.get('network', {})
        self.id = network_data.get('id', self.id)
        self.setup(network_data.get('input-count', 1),
                   network_data.get('hidden-layer_count', 1))
        self.set_input_values(network_data.get('input-values', [0]))
        self.set_perceptrons_per_layer(
            network_data.get('perceptrons-per-layer', [1]))
        for line in network_data.get('perceptrons_data', []):
            func_in_line = network_data['perceptrons_data'][line]['perceptons_activation_functions']
            weights_in_line = network_data['perceptrons_data'][line]['perceptons_weights']
            layer = self.get_layer_by_index(int(line))
            layer.set_children_functions_by_list(func_in_line)
            layer.set_children_weights(weights_in_line)

    def __repr__(self) -> str:
        return self.id


if __name__ == '__main__':

    network = NeuralNetwork(learning_rate=0.3, momentum=0.9)
    network.setup(4, 3)
    network.set_perceptrons_per_layer([3, 3, 1])
    # network.setup(2, 3)
    # network.set_perceptrons_per_layer([2, 2, 1])
    network.set_layer_activation_function(1, ActivationFunctions.SIGMOID_UNIPOLAR)
    network.set_layer_activation_function(2, ActivationFunctions.SIGMOID_UNIPOLAR)
    network.set_layer_activation_function(3, ActivationFunctions.SIGMOID_BIPOLAR)
    network.randomize_weights()
    network.train_backpropagation('resources\\training\\iris.csv',
                                  limit_iter=40_000,
                                  limit_time_sec=None,
                                  error_threshold=0.00001)
    # network.train_backpropagation('resources\\training\\iris.csv',
    #                               limit_iter=10000,
    #                               limit_time_sec=None,
    #                               error_threshold=1e-5)
    network.test([4.8,3.4,1.6,0.2]) #1
    network.test([5.0,3.3,1.4,0.2]) #1
    network.test([7.0,3.2,4.7,1.4]) #0
    network.test([5.8,2.7,5.1,1.9]) #0

    # network.test([0, 0])
    # network.test([0, 1])
    # network.test([1, 0])
    # network.test([1, 1])

