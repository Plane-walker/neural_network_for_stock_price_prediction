import tensorflow as tf
from tensorflow.keras.layers import Dropout, Dense, GRU
import matplotlib.pyplot as plt
from import_data import import_data


def main():
    train_input, test_input, train_output, test_label = import_data()
    print(train_input.shape, train_output.shape)
    model = tf.keras.Sequential([
        GRU(80, return_sequences=True),
        Dropout(0.2),
        GRU(100),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                  loss='mean_squared_error')
    history = model.fit(train_input, train_output, batch_size=64, epochs=50)
    predict_label = model.predict(test_input)
    plt.plot(test_label, color='red', label='Stock Price')
    plt.plot(predict_label, color='blue', label='Predicted Stock Price')
    plt.title('MaoTai Stock Price Prediction')
    plt.xlabel('Test sample')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
