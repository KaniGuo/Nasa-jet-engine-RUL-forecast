import pandas as pd
import numpy as np
import os
import pickle
from keras.models import load_model
import matplotlib.pyplot as plt


with open('Data_125_seq/x_test_FD001_seq30_rul125.pkl', 'rb') as file:      
    # Call load method to deserialze
    x_test = pickle.load(file)

rul001 = r'Data\RUL_FD001.txt'
rul_fd1 = pd.read_csv(rul001, sep=' ', header=None)
rul_fd1.drop(1, axis=1, inplace=True)

y_test = (rul_fd1.values).reshape(100,1)

sequence_length=30
#%%
file001_test = r'Data\test_FD001.txt'
fd001_test = pd.read_csv(file001_test, sep=' ', header=None)
# %%

num_cyc = []
index_x = []
row_u = -1
for i in range (1,101):
    print(i)
    cyc = len(np.where(fd001_test.iloc[:,0]==i)[0])
    num_cyc.append(cyc)
    delta_u = cyc-sequence_length
    row_u += delta_u
    index_x.append(row_u)
print(num_cyc)
print(index_x)
print(len(index_x))
# %%
# x_test = []
# y_test = []
# index_x = []
# u = 0
# for i in range(fd001_test.shape[0]-sequence_length):
#     if fd001_test.iloc[i, 0] == fd001_test.iloc[i+sequence_length, 0]:
#         print(fd001_test.iloc[i, 0])
#         u+=1
#         # x.append(fd001_test.iloc[i:i+sequence_length, 2:])
#     else:
#         print(u)
#         index_x.append(u)
# print(u)

# unique_list = list(set(index_x))

# print(sorted(unique_list))
# print(len(unique_list))

# %%
extracted_data_list = [x_test[index, :, :] for index in index_x]

x_pred = np.array(extracted_data_list)

print(x_pred.shape) 
# %%
model = load_model('GRU_results/GRU_3_128_64_32_Dense_2_64_1_AdamW_seq30_125_bs256/model.h5')
y_pred = model.predict(x_pred)
y_pred=y_pred.reshape(100,)
y_test=y_test.reshape(100,)
# %%
plt.figure()
plt.plot(y_test.reshape(-1, 1), label='Actual')
plt.plot(y_pred, label='Predicted')
plt.xlabel('Engine Unit')
plt.ylabel('RUL')
plt.legend()
# plt.savefig(os.path.join(new_folder_path, 'value_plot.png'))
plt.show(y_pred.shape)
print
# %%
sorted_indices = np.argsort(y_test)
sorted_y_pred = y_pred[sorted_indices]
sorted_y_test = y_test[sorted_indices].reshape(-1)

# 
plt.figure(figsize=(10, 6))
plt.plot(sorted_y_test, label='Actual')
plt.plot(sorted_y_pred, label='Predicted')
plt.xlabel('Engine Unit')
plt.ylabel('RUL')
plt.legend()
plt.title('Comparison of Actual and Predicted Values')
plt.savefig('predict_actual.png')
plt.show()
# %%
