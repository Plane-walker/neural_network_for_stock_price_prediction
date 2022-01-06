import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import Dropout, Dense, GRU
import matplotlib.pyplot as plt
from import_data import import_data


def main():
    x_train, x_val, x_test, x_long_term, y_train, y_val, y_test, y_long_term = import_data()
    model = tf.keras.Sequential([
        GRU(80, return_sequences=True),
        Dropout(0.2),
        GRU(120, return_sequences=True),
        Dropout(0.2),
        GRU(100),
        Dense(1)
    ])
    model.compile(optimizer='rmsprop',
                  loss='mean_squared_error')
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), batch_size=64, epochs=50)
    y_predict = model.predict(x_test)
    test_loss = model.evaluate(x_test, y_test)
    plt.plot(history.history['loss'], label='MSE (training data)')
    plt.plot(history.history['val_loss'], label='MSE (validation data)')
    plt.axhline(test_loss, linestyle='--', label='MSE (test data)')
    plt.title('MSE for Stock Price Prediction')
    plt.ylabel('MSE value')
    plt.xlabel('No. epoch')
    plt.legend(loc="upper left")
    plt.show()
    plt.plot(y_test, label='Stock Price')
    plt.plot(y_predict, label='Predicted Stock Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Test sample')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
    y_predict_long_term = model.predict(x_long_term)
    for y_index in y_long_term:
        x_long_term = np.delete(x_long_term, 0, axis=1)
        x_long_term = np.insert(x_long_term, x_long_term.shape[1], values=[y_predict_long_term[0, -1],
                                                                           y_predict_long_term[0, -1],
                                                                           y_predict_long_term[0, -1],
                                                                           y_predict_long_term[0, -1],
                                                                           x_long_term[0, -1, 4],
                                                                           x_long_term[0, -1, 5],
                                                                           x_long_term[0, -1, 6],
                                                                           x_long_term[0, -1, 7],
                                                                           x_long_term[0, -1, 8],
                                                                           x_long_term[0, -1, 9]], axis=1)
        y_predict_long_term = np.insert(y_predict_long_term, y_predict_long_term.shape[0], values=[model.predict(x_long_term)[0, 0]], axis=0)
    plt.plot(y_long_term, label='Stock Price')
    plt.plot(y_predict_long_term, label='Predicted Stock Price')
    plt.title('Stock Price Prediction')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()
