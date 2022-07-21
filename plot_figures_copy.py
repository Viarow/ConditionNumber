import numpy as np 
import matplotlib.pyplot as plt

# SER: tab:blue, tab:green, tab:orange, tab:red

SNRdB_range = np.arange(0, 4, 0.5)
num_mods = 3
#labels = ['maximum likelihood', 'MMSE', 'Langevine', 'MMNet']
# colors = ['tab:blue', 'tab:green', 'tab:orange', 'tab:red']
# markers = ['o', 'v', '*', 'x']
labels = ['cond=2.24', 'cond=2.57', 'cond=3.64']
colors = ['tab:blue', 'tab:green', 'tab:orange']
markers = ['o', 'v']

SER_data = np.array([
    [0.01835, 0.01445, 0.01045, 0.00735, 0.00505, 0.00315, 0.00205, 0.00155],
    [1.7e-03, 8.0e-04, 5.0e-04, 2.5e-04, 5.0e-05, 0.0e+00, 0.0e+00, 0.0e+00],
    [2.5e-04, 1.0e-04, 5.0e-05, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00]
])

for i in range(0, num_mods):
    for j in range(0, SNRdB_list.shape[0]):
        if SER_data[i][j] == 0. :
            SER_data[i][j] = np.nan

for i in range(0, num_mods):
    plt.plot(SNRdB_list, SER_data[i], color=colors[i], marker=markers[i], markersize=7, label=labels[i])

#plt.xlim(-0.5, 16.5)
plt.ylim(1e-4, 1)
plt.yscale('log')
plt.xlabel('SNR')
plt.ylabel('Symbol Error Rate')
plt.legend(loc='lower left', fontsize = 'x-small')
plt.grid(True)
plt.savefig('./figures/Comp_16to16/ResNet_like.png')