import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def import_data_from_origin_file():
    data = pd.read_excel('2020A.xlsx').values
    stock_ids = np.unique(data[:, 2])
    inputs = []
    labels = []
    for stock_id in stock_ids:
        stock_time_line = data[data[:, 2] == stock_id]
        for index in range(stock_time_line.shape[0] // 31):
            inputs.extend(stock_time_line[index * 31: index * 31 + 30, 17:])
            labels.extend([[stock_time_line[index * 31 + 30, 17]]])
    inputs = np.array(inputs)
    labels = np.array(labels)
    pd.DataFrame(inputs).to_csv('inputs.csv', index=False, header=False)
    pd.DataFrame(labels).to_csv('labels.csv', index=False, header=False)


def import_data():
    inputs = pd.read_csv('inputs.csv', sep=',', header=None).values
    labels = pd.read_csv('labels.csv', sep=',', header=None).values
    print(labels[(labels[:, 0] > 1500)])
    inputs = inputs.reshape(-1, 30, 10)
    return train_test_split(inputs, labels, test_size=0.2)


if __name__ == '__main__':
    import_data()
