import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def import_data_from_origin_file():
    data = pd.read_excel('2020A.xlsx').values
    stock_ids = np.unique(data[:, 2])
    inputs = []
    labels = []
    for stock_id in stock_ids:
        stock_time_line = data[data[:, 2] == stock_id]
        #stock_time_line是每支股票的全年大概242天的相关记录，包括开盘，收盘等等
        temp_list = []
        count = 0
        for i in stock_time_line[:,17:]:
            for j in i:
                if j < 0:
                    temp_list.append(count)
                    temp_bool = False
                    break
            count += 1
        temp_result = np.delete(stock_time_line[:,17:], temp_list, axis=0)

        temp_bool = True
        for i in temp_result:
            for j in i[:4]:
                if j > 100:
                    temp_bool = False
                    break

        if temp_bool:
            # 停牌的股票这一项为444016000，未停牌的股票为这一项0。
            for index in range(len(temp_result) // 31):
                inputs.extend(temp_result[index * 31: index * 31 + 30][:4])
                labels.extend([[temp_result[index * 31 + 30][0]]])
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
    # if not os.path.exists('inputs.csv'):
    #     import_data_from_origin_file()
    # import_data()
    import_data_from_origin_file()
