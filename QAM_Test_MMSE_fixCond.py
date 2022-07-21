import numpy as np 
from numpy.linalg import cond 
import time
from CommPy.utils import *
from CommPy.modulation import QAMModem
from CommPy.detector import MMSE_general
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

""" Global Variables"""
N = 16                 # number of users
M = 16                # number of receiving antennas
block_size = 500
mod = 16               # modulation order, for BPSK is 2
bit_width = int(np.log2(mod))  # number of bits in one symbol
SNRdB_range = np.arange(0, 18, 2)
H_mean = 0            # channel matrix distribution
H_var = 1             # channel matrix distribution
load_path = './data/QAM4_16to16/channel_matrices.npy'
fig_save_path = './figures/QAM16_16to16/multi_users/MMSE_Cond84_Multi.png'


""" TODO: Main Change Here """
def single_time_transmit(x_symbols, signal_power, var_noise, H, constellation):
    # compute noise
    noise_real = 0. + np.sqrt(0.5 * var_noise)*np.random.randn(M)
    noise_imag = 0. + np.sqrt(0.5 * var_noise)*np.random.randn(M)

    # channel: Rayleigh Fading
    y_symbols = np.matmul(H, x_symbols) + (noise_real + 1j * noise_imag)

    """ detection """
    lamda = var_noise/signal_power
    xhat_symbols = MMSE_general(H, y_symbols, lamda, constellation)
    
    return xhat_symbols


def plot_figure(SNRdB_range, SER_data, fig_save_path):
    sns.set_style("whitegrid")
    sns.set_palette(sns.color_palette("Paired", n_colors=N))

    # adjust 0 to np.nan
    for i in range(0, N):
        for j in range(0, SNRdB_range.shape[0]):
            if SER_data[i][j] == 0. :
                SER_data[i][j] = np.nan

    for i in range(0, N):
        plt.plot(SNRdB_range, SER_data[i], markersize=7, label='user{:d}'.format(i+1))
    
    plt.xlim(-0.5, 16.5)
    plt.ylim(1e-4, 1)
    plt.yscale('log')
    plt.xlabel('SNRdB')
    plt.ylabel('Symbol Error Rate')
    plt.legend(loc='lower left', fontsize = 'x-small', ncol=4)
    plt.grid(True)
    plt.savefig(fig_save_path)


def main():

    # create modem
    modem = QAMModem(mod)
    constellation = modem.constellation

    # compute signal power
    signal_power = np.abs(constellation[0])

    # record BER after decoding and CRC pass rate
    SER_uncoded_main = np.zeros([N, SNRdB_range.shape[0]])
    SER_all = np.zeros([SNRdB_range.shape[0], ])

    H_list = np.load(load_path, allow_pickle=True)
    idx = 85
    H = H_list[idx, :, :]
    print(cond(H))

    start_time = time.time()
    """MAIN LOOPS"""
    for dd in range(0, SNRdB_range.shape[0]):
        var_noise = signal_power * H_var * np.power(10, -0.1*SNRdB_range[dd])
        SER_uncoded_block = np.zeros([N, block_size])

        for bb in tqdm(range(0, block_size)):
            # data bits in one packet
            x_bits = np.random.randint(0, 2, (bit_width*N, ))
            # no encoding
            x_indices, x_symbols = modem.modulate(x_bits)
            # transmit, detect
            xhat_symbols = single_time_transmit(x_symbols, signal_power, var_noise, H, constellation)
            # demodulate
            demod_bits = modem.demodulate(xhat_symbols, 'hard')
            # get xhat_indices
            xhat_indices, _ = modem.modulate(demod_bits)
            # SER
            SER_uncoded_block[:, bb] = symbol_err_rate(x_indices, xhat_indices)

        # iteration over a block ends here
        SER_uncoded_main[:, dd] = np.mean(SER_uncoded_block, axis=1)
        SER_all[dd] = np.sum(SER_uncoded_block) / (N * block_size)

        print("--- dd=%d --- SNR = %.1f dB  ---" % (dd, SNRdB_range[dd]))
        print(" MMSE Symbol Error Rate Of Each User: ")
        print(SER_uncoded_main)
        print(" Overall Symbol Error Rate: ")
        print(SER_all)
    
    # iteration over all SNR ends here
    running_time = time.time() - start_time
    print('Running finished in %s seconds.'% (running_time))

    plot_figure(SNRdB_range, SER_uncoded_main, fig_save_path)


if __name__ == '__main__':
    main()