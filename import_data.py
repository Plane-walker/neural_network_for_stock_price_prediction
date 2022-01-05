import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def import_data_from_origin_file():
    data = pd.read_excel('2020A.xlsx').values
    stock_ids = np.unique(data[:, 2])
    np.random.shuffle(stock_ids)
    inputs = []
    labels = []
    long_term_inputs = []
    long_term_labels = []
    count = 0
    for stock_id in stock_ids:
        stock_time_line = (data[data[:, 2] == stock_id])[:, 17:]
        abnormal_index = np.unique(np.concatenate((np.argwhere(stock_time_line < 0)[:, 0], np.argwhere(stock_time_line[:, 0: 4] > 200)[:, 0]), axis=0))
        drop = False
        for index in abnormal_index:
            if index == 0 or index == stock_time_line.shape[0] - 1:
                continue
            if np.any(stock_time_line[index, 0: 4] > 200):
                drop = True
                break
            if np.any(stock_time_line[index - 1] < 0) or np.any(stock_time_line[index + 1] < 0):
                drop = True
                break
            stock_time_line[index] = (stock_time_line[index - 1] + stock_time_line[index + 1]) / 2
        if drop:
            continue
        if np.any(stock_time_line[0] < 0):
            stock_time_line = np.delete(stock_time_line, 0, axis=0)
        if np.any(stock_time_line[stock_time_line.shape[0] - 1] < 0):
            stock_time_line = np.delete(stock_time_line, stock_time_line.shape[0] - 1, axis=0)
        if count == 0 and stock_time_line.shape[0] // 31 > 1:
            long_term_inputs.extend(stock_time_line[0: 30, :])
            for index in range(30, 61):
                long_term_labels.extend([[stock_time_line[index, 0]]])
            count += 1
        else:
            for index in range(stock_time_line.shape[0] // 31):
                inputs.extend(stock_time_line[index * 31: index * 31 + 30, :])
                labels.extend([[stock_time_line[index * 31 + 30, 0]]])
    inputs = np.array(inputs)
    labels = np.array(labels)
    pd.DataFrame(inputs).to_csv('inputs.csv', index=False, header=False)
    pd.DataFrame(labels).to_csv('labels.csv', index=False, header=False)
    pd.DataFrame(long_term_inputs).to_csv('long_term_inputs.csv', index=False, header=False)
    pd.DataFrame(long_term_labels).to_csv('long_term_labels.csv', index=False, header=False)


def import_data():
    inputs = pd.read_csv('inputs.csv', sep=',', header=None).values
    labels = pd.read_csv('labels.csv', sep=',', header=None).values
    long_term_x = pd.read_csv('long_term_inputs.csv', sep=',', header=None).values
    long_term_y = pd.read_csv('long_term_labels.csv', sep=',', header=None).values
    inputs = inputs.reshape(-1, 30, 10)
    long_term_x = long_term_x.reshape(-1, 30, 10)
    train_ratio = 0.75
    validation_ratio = 0.15
    test_ratio = 0.10
    x_train, x_test, y_train, y_test = train_test_split(inputs, labels, test_size=1 - train_ratio)
    x_val, x_test, y_val, y_test = train_test_split(x_test, y_test, test_size=test_ratio/(test_ratio + validation_ratio))
    return x_train, x_val, x_test, long_term_x, y_train, y_val, y_test, long_term_y


if __name__ == '__main__':
    import_data_from_origin_file()
