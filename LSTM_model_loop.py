#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
import tensorflow as tf
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
import os

#%%
r2_data = pd.DataFrame(columns=[64, 128, 256], index=[10, 13, 20, 26, 30, 35, 40, 50, 52, 55, 60])

for seq_len in [10, 13, 20, 26, 30, 35, 40, 50, 52, 55, 60]:
  for batch in [64,128, 256]:
        batch_size = batch # 64 128 256
        seq = seq_len

        print('-'*25)
        print(f'Batch_size: {batch}, Sequence: {seq}')
        print('-'*25)

        with open(f'x_train_FD001_seq{seq}_rul125.pkl', 'rb') as file:      
            # Call load method to deserialze
            x_train = pickle.load(file)

        with open(f'y_train_FD001_seq{seq}_rul125.pkl', 'rb') as file:      
            # Call load method to deserialze
            y_train = pickle.load(file)

        with open(f'x_test_FD001_seq{seq}_rul125.pkl', 'rb') as file:      
            # Call load method to deserialze
            x_test = pickle.load(file)

        with open(f'y_test_FD001_seq{seq}_rul125.pkl', 'rb') as file:      
            # Call load method to deserialze
            y_test = pickle.load(file)

        print(x_train.shape)
        print(y_train.shape)
        print(x_test.shape)
        print(y_test.shape)
        # %%

        sequence_length = seq
        num_features = 14

        model = keras.models.Sequential()
        model.add(keras.layers.LSTM(input_shape=(sequence_length, num_features),
                    units=128,
                    dropout=0.1,
                    return_sequences=True))
        model.add(keras.layers.LSTM(input_shape=(sequence_length, num_features),
                    units=64,
                    dropout=0.1,
                    return_sequences=True))
        model.add(keras.layers.LSTM(input_shape=(sequence_length, num_features),
                    units=32,
                    dropout=0.1,
                    return_sequences=False))
        model.add(keras.layers.Dense(units=64, activation='relu'))
        # model.add(keras.layers.Dense(units=32, activation='relu'))
        model.add(keras.layers.Dense(units=1, activation='linear'))

        optimizer = keras.optimizers.experimental.AdamW(learning_rate=0.001)
        # optimizer = keras.optimizers.SGD() # very bad performance 
        model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

        model.summary()

        #%%

        opt = str(optimizer.__class__).split('.')[-1][:-2]
        n_lstm = sum(['lstm' in str(i.__class__) for i in model.layers])
        n_dense = sum(['dense' in str(i.__class__) for i in model.layers])

        folder_name = f'LSTM_{n_lstm}_128_64_32_Dense_{n_dense}_64_1_{opt}_seq{sequence_length}_125_bs{batch_size}'

        path = Path(r'D:\4.Learning\ML_AI\Alfatraining\DeepLearning\PredictiveMaintenance\NASA-turbofan-jet-engine')
        
        output_path = Path(path / 'Results' / folder_name)
        
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        with open(os.path.join(output_path, 'model_summary.txt'), 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))


        # %%
        lr_reduce = keras.callbacks.ReduceLROnPlateau(patience=5)
        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', 
                                                min_delta=0, 
                                                patience=10,
                                                verbose=1,
                                                restore_best_weights=False)

        history = model.fit(x_train, y_train, epochs=50, batch_size=batch_size,
                            validation_data=(x_test, y_test),
                            callbacks=[lr_reduce, early_stop])
        # stopped_epoch = history.stopped_epoch
        # %%
        model.save(output_path / 'model.h5')

        #%%
        # draw the epoch-loss
        def loss_plot(save):
            plt.figure()
            plt.plot(history.history['loss'], label='Train Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            if save==True:
                plt.savefig(os.path.join(output_path, 'loss_plot.png'))
            # plt.show()

        # epoch-accuracy
        def accuracy_plot():
            plt.plot(history.history['Accuracy'], label='Train Accuracy')
            plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            plt.show()

        # epoch-R2 
        def r2_plot():
            plt.plot(history.history['r2_score'], label='Train R2 Score')
            plt.plot(history.history['val_r2_score'], label='Validation R2 Score')
            plt.xlabel('Epoch')
            plt.ylabel('R2 Score')
            plt.legend()
            plt.show()

        def mae_plot(save=False):
            plt.figure()
            plt.plot(history.history['mae'], label='Train MAE')
            plt.plot(history.history['val_mae'], label='Validation MAE')
            plt.xlabel('Epoch')
            plt.ylabel('MAE')
            plt.legend()
            if save == True:
                plt.savefig(os.path.join(output_path, 'mae_plot.png'))
            # plt.show()

        mae_plot(save=True)

        #%%
        # Plotting 
        loss_plot(save=True)

        # %%
        print('-'*25)
        print(" Predicting ".center(20, '#'))
        print('-'*25)
        y_pred = model.predict(x_test)
        #%%
        print(x_train.shape)
        print(x_test.shape)
        print(y_pred.shape)
        #%%
        def value_plot(save=False):
            plt.figure()
            plt.plot(y_test.reshape(-1, 1), label='Actual')
            plt.plot(y_pred, label='Predicted')
            plt.xlabel('Data point')
            plt.ylabel('Value')
            plt.legend()
            if save == True:
                plt.savefig(os.path.join(output_path, 'value_plot.png'))
            # plt.show()

        value_plot(save=True)

        # %%
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print("Mean Squared Error:", mse)
        print("Mean Absolute Error:", mae)
        print("R2 Score:", r2)

        with open(os.path.join(output_path, 'r2_scores.txt'), 'w') as f:
            f.write(f'R2 Score: {r2}\n')
            f.write(f'Mean Squared Error: {mse}\n')
            f.write(f'Mean Absolute Error: {mae}')




        # %%
        # Saving R2 data
        r2_data.loc[seq, batch_size] = r2

r2_data.to_csv(Path(path / 'Results') / 'r2_scores.csv')
