import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

# SER: tab:blue, tab:green, tab:orange, tab:red

SNRdB_range = SNRdB_range = np.arange(0, 11, 1)
num_methods = 8
labels = ['MMSE_center1.5', 'MMSE_center2.0', 'MMSE_center2.5', 'MMSE_center3.0', 'Langevine_center1.5', 'Langevine_center2.0', 'Langevine_center2.5', 'Langevine_center3.0']
colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red', 'tab:blue', 'tab:green', 'tab:orange', 'tab:red']
markers = ['o', 'o', 'o', 'o', '*', '*', '*', '*']
# labels = ['MMSE', 'Langevine', 'MMNet']
# colors = ['tab:green', 'tab:orange', 'tab:red']
# markers = ['v', '*', 'x']


SER_data = np.array([
    [0.0182, 0.0083, 0.00245, 0.000725, 0.0001,  0.,  0.,  0., 0.,  0., 0.],
    [0.043675, 0.021975, 0.009925, 0.0041,  0.00125,  0.00035,  0.0001, 0., 0., 0., 0.],
    [5.1975e-02, 2.7275e-02, 1.2925e-02, 5.5500e-03, 1.4000e-03, 5.5000e-04, 1.5000e-04, 2.5000e-05, 0.00e+00, 0.00e+00, 0.00e+00],
    [0.084875, 0.052525, 0.026425, 0.013275, 0.00585,  0.001925, 0.00055,  0.0002,  0., 0., 0.],
    [8.50e-04, 2.75e-04, 2.50e-05, 2.50e-05, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00],
    [1.725e-03, 6.750e-04, 7.500e-05, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00],
    [4.325e-03, 1.200e-03, 3.750e-04, 7.500e-05, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00],
    [3.500e-03, 1.825e-03, 3.250e-04, 2.000e-04, 2.500e-05, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00, 0.00e+00]
])

for i in range(0, num_methods):
    for j in range(0, SNRdB_range.shape[0]):
        if SER_data[i][j] == 0. :
            SER_data[i][j] = np.nan

#sns.set_style("whitegrid")

for i in range(0, 4):
    plt.plot(SNRdB_range, SER_data[i], linestyle='dashed', color=colors[i], marker=markers[i], markersize=6, label=labels[i])
for i in range(4, num_methods):
    plt.plot(SNRdB_range, SER_data[i], color=colors[i], marker=markers[i], markersize=6, label=labels[i])

plt.xlim(0, 10)
plt.ylim(1e-5, 1)
plt.yscale('log')
plt.xlabel('SNR(dB)')
plt.ylabel('Symbol Error Rate')
plt.legend(loc='upper right', fontsize = 'x-small', ncol=2)
plt.grid(True)
plt.savefig('./figures/QAM16_4to16/MIMO_MMSE_vs_Langevine.png')