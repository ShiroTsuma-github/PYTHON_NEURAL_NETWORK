import numpy as np
import csv
import sklearn.neural_network

data = []
output = []
with open('resources\\training\\iris.csv', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        if len(row) == 0:
            continue
        data.append([float(val) for val in row[:-1]])
        output.append(float(row[-1]))


inputs = np.array(data)
expected_output = np.array(output)

model = sklearn.neural_network.MLPClassifier(
                activation='logistic',
                max_iter=100_000,
                hidden_layer_sizes=(2),
                solver='adam')
model.fit(inputs, expected_output)
results = model.predict(inputs)
for i in range(len(results)):
    if results[i] != expected_output[i]:
        print('Error: expected:', expected_output[i], 'actual:', results[i])
    print('expected:', expected_output[i], 'actual:', results[i])
# print('predictions:', model.predict(inputs))