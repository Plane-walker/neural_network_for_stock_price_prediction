import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
from icecream import ic

def import_data_from_origin_file():
    data = pd.read_excel('2020A.xlsx').values
    temp_data = np.unique(data[:, 2])
    stock_ids = temp_data[temp_data!='600519.SH']
    inputs = []
    labels = []
    for stock_id in stock_ids:
        stock_time_line = data[data[:, 2] == stock_id]
        if stock_time_line[:,17:].all() > 0 and stock_time_line[0,14] == 0 and stock_time_line[:,17:21].max() < 100:
            # 停牌的股票这一项为444016000，未停牌的股票为这一项0。
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
    if not os.path.exists('inputs.csv'):
        import_data_from_origin_file()
    import_data()