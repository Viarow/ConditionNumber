import numpy as np 
from numpy.linalg import cond 
import time
import random
from CommPy.utils import *
from CommPy.modulation import QAMModem
from CommPy.detector import MMSE_general
import matplotlib.pyplot as plt
from tqdm import tqdm

""" Global Variables"""
N = 4                 # number of users
M = 16                # number of receiving antennas
block_size = 1000
mod = 16               # modulation order, for BPSK is 2
bit_width = int(np.log2(mod))  # number of bits in one symbol
SNRdB_range = np.arange(0, 11, 1)
H_mean = 0            # channel matrix distribution
H_var = 1             # channel matrix distribution
data_load_path = './data/QAM16_4to16/channel_matrices.npy'
idx_load_path = './data/QAM16_4to16/matrix_slices.npy'
num_H = 10


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


def main():

    # create modem
    modem = QAMModem(mod)
    constellation = modem.constellation

    # compute signal power
    signal_power = np.abs(constellation[0])

    # record BER after decoding and CRC pass rate
    SER_uncoded_main = np.zeros([num_H, SNRdB_range.shape[0]])

    H_list = np.load(data_load_path, allow_pickle=True)
    idx_slice = np.load(idx_load_path, allow_pickle=True).item().get('center3.0')
    H_idx = random.sample(idx_slice, num_H)

    start_time = time.time()

    for i in range(0, num_H):
        H = H_list[H_idx[i], :, :]
        print(cond(H))

        """MAIN LOOPS"""
        for dd in range(0, SNRdB_range.shape[0]):
            var_noise = signal_power * H_var * np.power(10, -0.1*SNRdB_range[dd])
            SER_uncoded_block = np.zeros([block_size, ])

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
                SER_uncoded_block[bb] = bit_err_rate(x_indices, xhat_indices)

            # iteration over a block ends here
            SER_uncoded_main[i, dd] = np.mean(SER_uncoded_block)

            #print("--- dd=%d --- SNR = %.1f dB  ---" % (dd, SNRdB_range[dd]))
            #print(" MMSE Symbol Error Rate: ")
            #print(SER_uncoded_main)
    
        # iteration over all SNR ends here
        # running_time = time.time() - start_time
        # print('Running for cond = %.1f finished in %s seconds.'% (cond(H), running_time))

    # iteration over all H matrices ends here
    SER_averge_main = np.mean(SER_uncoded_main, axis = 0)
    running_time = time.time() - start_time
    print('Running for all H matrices finished in %s seconds.'% (running_time))
    print(SER_averge_main)

if __name__ == '__main__':
    main()