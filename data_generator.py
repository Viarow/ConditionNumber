import numpy as np
from numpy.linalg import cond 
import matplotlib.pyplot as plt

N = 4            # number of transmitting antennas or number of users
M = 16            # number of receiving antennas
H_mean = 0       # Rayleigh Fading H
H_var = 1        # Rayleigh Fading H
# try 1000 H to see the distribution
num_H = 1000
# path to store the data
data_store_path = './data/QAM16_4to16/channel_matrices.npy'
idx_store_path = './data/QAM16_4to16/matrix_slices.npy'

raw_H = np.sqrt(0.5 * H_var) * (np.random.randn(num_H, M, N) + 1j * np.random.randn(num_H, M, N))
raw_cond = np.zeros([num_H, ])
for i in range(0, num_H):
    single_H = raw_H[i, :, :]
    raw_cond[i] = cond(single_H)

sorted_idx = np.argsort(raw_cond)
ordered_H = np.zeros(raw_H.shape) + 1j * np.zeros(raw_H.shape)
ordered_cond = np.zeros(raw_cond.shape)
for i in range(0, num_H):
    ordered_H[i, :, :] = raw_H[sorted_idx[i], :, :]
    ordered_cond[i] = raw_cond[sorted_idx[i]]

list_1 = []
list_2 = []
list_3 = []
list_4 = []
for i in range(0, num_H):
    if (np.abs(ordered_cond[i])-1.5 < 0.1):
        list_1.append(i)
    elif (np.abs(ordered_cond[i])-2 < 0.1):
        list_2.append(i)
    elif (np.abs(ordered_cond[i])-2.5 < 0.1):
        list_3.append(i)
    elif (np.abs(ordered_cond[i])-3 < 0.1):
        list_4.append(i)

print(list_1)
print(len(list_1))
print(list_2)
print(len(list_2))
print(list_3)
print(len(list_3))
print(list_4)
print(len(list_4))

slices_dict = {
    'center1.5': list_1,
    'center2.0': list_2,
    'center2.5': list_3,
    'center3.0': list_4
}

np.save(data_store_path, ordered_H)
np.save(idx_store_path, slices_dict)
#print(ordered_cond)
plt.hist(ordered_cond, bins='auto')
#plt.xscale('log')
plt.gca().set(title='Condition Number Histogram', ylabel='Frequency')
plt.savefig('./data/QAM16_4to16/ConditionNumber_Histogram.png')