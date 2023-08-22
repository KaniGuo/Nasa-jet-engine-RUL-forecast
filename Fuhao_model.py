#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
import tensorflow as tf
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os



#%%
def load_data(path):
    with open(f'{path}/x_train_FD001.pkl', 'rb') as file:      
        # Call load method to deserialze
        x_train = pickle.load(file)

    with open(f'{path}/y_train_FD001.pkl', 'rb') as file:      
        # Call load method to deserialze
        y_train = pickle.load(file)

    with open(f'{path}/x_test_FD001.pkl', 'rb') as file:      
        # Call load method to deserialze
        x_test = pickle.load(file)

    with open(f'{path}/y_test_FD001.pkl', 'rb') as file:      
        # Call load method to deserialze
        y_test = pickle.load(file)

    return x_train, y_train, x_test, y_test

sequence_lengths = [50]
for sequence_length in sequence_lengths:
    path_data = str(sequence_length)
    x_train, y_train, x_test, y_test = load_data(path_data)
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
# %%

    # sequence_length = 26
    print(sequence_length)
    num_features = 14
    model = keras.models.Sequential()
    model.add(keras.layers.GRU(input_shape=(sequence_length, num_features),
                units=128,
                dropout=0.1,
                return_sequences=True))
    model.add(keras.layers.GRU(input_shape=(sequence_length, num_features),
                units=64,
                dropout=0.1,
                return_sequences=True))
    model.add(keras.layers.GRU(input_shape=(sequence_length, num_features),
                units=32,
                dropout=0.1,
                return_sequences=False))
    model.add(keras.layers.Dense(units=64, activation='relu'))
    # model.add(keras.layers.Dense(units=32, activation='relu'))
    model.add(keras.layers.Dense(units=1, activation='linear'))

    optimizer = keras.optimizers.AdamW(learning_rate=0.001)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

    model.summary()
    # %%
    lr_reduce = keras.callbacks.ReduceLROnPlateau(patience=5)

    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=10, restore_best_weights=True)

    history = model.fit(x_train, y_train, epochs=50, batch_size=128,
                        validation_data=(x_test, y_test),
                        # callbacks=[lr_reduce])
                        callbacks=[lr_reduce, early_stop])

    # stopped_epoch = early_stop.stopped_epoch
    # print("Model stopped training at epoch:", stopped_epoch)

    # %%

    output_parent_folder = 'Result'

    # get the list of folder
    existing_folders = os.listdir(output_parent_folder)
    existing_indices = [int(folder_name) for folder_name in existing_folders if folder_name.isdigit()]

    # calculate the new number for new folder
    if existing_indices:
        new_folder_index = max(existing_indices) + 1
    else:
        new_folder_index = 1

    # create new folder
    new_folder_path = os.path.join(output_parent_folder, str(new_folder_index))
    os.makedirs(new_folder_path, exist_ok=True)





    # def create_folder(folder_path):
    #     if not os.path.exists(folder_path):
    #         os.makedirs(folder_path)

    # output_folder = 'Result/training5' 
    # create_folder(output_folder)

    model.save(new_folder_path+'/my_model.h5')

    print("Done!")
    #%%
    # draw the epoch-loss
    plt.figure()
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(new_folder_path, 'loss_plot.png'))
    # plt.show()

    #%%
    plt.figure()
    plt.plot(history.history['mae'], label='Train MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    plt.savefig(os.path.join(new_folder_path, 'MAE_plot.png'))
    # plt.show()
    # %%
    y_pred = model.predict(x_test)
    # %%
    plt.figure()
    plt.plot(y_test.reshape(-1, 1), label='Actual')
    plt.plot(y_pred, label='Predicted')
    plt.xlabel('Data point')
    plt.ylabel('Value')
    plt.legend()
    plt.savefig(os.path.join(new_folder_path, 'value_plot.png'))
    # plt.show()
    # %%
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Mean Squared Error:", mse)
    print("Mean Absolute Error:", mae)
    print("R2 Score:", r2)

    # %%
    # 保存模型摘要

    with open(os.path.join(new_folder_path, 'model_summary.txt'), 'w') as f:
        # 使用 file 参数将输出写入文件
        model.summary(print_fn=lambda x: f.write(x + '\n'))

    with open(os.path.join(new_folder_path, 'r2_scores.txt'), 'w') as f:
        f.write(f'R2 Score: {r2}\n')
        f.write(f'Mean Squared Error: {mse}\n')
        f.write(f'Mean Absolute Error: {mae}\n')
        # f.write(f"Model stopped training at epoch:", {stopped_epoch})
        f.write(f'Sequence length: {sequence_length}\n')
        # f.write(f"optimizer:", {optimizer}) 

