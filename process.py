import importlib
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.stats import median_abs_deviation
#import config_visual
import config
from scipy.signal import butter, lfilter


decoder = importlib.import_module(f'decoders.{config.decoder}')


def string_is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def lowpass(csi_vec: np.array, cutoff: float, fs: float, order: int) -> np.array:
    nyq = 0.5*fs
    normal_cutoff = cutoff/nyq
    b, a = signal.butter(order, normal_cutoff, btype="low", analog=False)
    return signal.filtfilt(b, a, csi_vec)


def hampel_filter(input_array, window_size, n_sigmas=3):
    length = len(input_array)
    new_array = input_array.copy()
    k = 1.4826  # scale factor for Gaussian distribution
    for i in range(window_size, length - window_size):
        local_window = input_array[i - window_size:i + window_size + 1]
        median = np.median(local_window)
        std = k * np.median(np.abs(local_window - median))
        if np.abs(input_array[i] - median) > n_sigmas * std:
            new_array[i] = median
    return new_array


def hampel(csi, k, nsigma):
    index = 0
    csi = csi.copy()
    for x in csi:
        y = 0
        if index <= k:
            #Special case, first few samples.
            y = k
        elif index+k > len(csi):
            #Special case, last few samples
            y = -k

        index += y
        stdev = np.std(csi[index-k:index+k])
        median = np.median(csi[index-k:index+k])
        index -= y

        if abs(x-median) > nsigma * stdev:
            csi[index] = median
        index += 1
    return csi


def validate_csi_packet(csi, x, index):
    for i in range(len(x) - 9):
        if np.all(np.abs(csi[i:i + 10]) < 50):
            print(f"Skipping packet {index} as there are 10 consecutive CSI values less than 50 starting at index {i}.")
            return None

    # If there are not 10 consecutive frequencies with a CSI value less than 50, return CSI data
    return csi

def plot_csi(samples, index, bandwidth, filename, valid_packets_indices):
    if index not in valid_packets_indices:
        print(f"Sample {index} is invalid and will not be plotted.")
        return
    csi = samples.get_csi(index, config.remove_null_subcarriers, config.remove_pilot_subcarriers)
    if index >= samples.nsamples:
        print(f"Index {index} is out of bounds. Total samples: {samples.nsamples}")
        return

    csi = samples.get_csi(index, config.remove_null_subcarriers, config.remove_pilot_subcarriers)
    if csi is None:
        print(f"No CSI data available for sample {index}.")
        return

    cutoff = 40
    fs = 100
    csi_lowpass_filtered = lowpass(csi, cutoff, fs, order=5)

    nsub = int(bandwidth * 3.2)
    x = np.arange(-1 * nsub / 2, nsub / 2)
    csi_abs = np.absolute( csi_lowpass_filtered)
    #csi_final=hampel_filter(csi_abs,5,3)

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(x,csi_abs, label=f'Amplitude Index {index}')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(x, np.angle(csi_lowpass_filtered, deg=True), label=f'Phase Index {index}')
    plt.ylabel('Phase')
    plt.xlabel('Subcarrier')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_csi_range(samples, start, end, bandwidth, filename, valid_packets_indices):
    nsub = int(bandwidth * 3.2)
    x = np.arange(-1 * nsub / 2, nsub / 2)

    plt.figure(figsize=(14, 6))
    ax_amp = plt.subplot(2, 1, 1)
    ax_pha = plt.subplot(2, 1, 2)

    for index in range(start, min(end + 1, samples.nsamples)):
        if index not in valid_packets_indices:
            print(f"Sample {index} is invalid and will not be plotted.")
            continue

        csi = samples.get_csi(index, config.remove_null_subcarriers, config.remove_pilot_subcarriers)
        cutoff = 40
        fs = 100
        csi_lowpass_filtered = lowpass(csi, cutoff, fs, order=5)
        samples.print(index)
        csi = samples.get_csi(index, config.remove_null_subcarriers, config.remove_pilot_subcarriers)
        csi_abs = np.absolute(csi_lowpass_filtered)
        if csi is None:
            continue

        ax_amp.plot(x, np.abs(csi), label=f'Sample {index}')
        ax_pha.plot(x, np.angle(csi, deg=True), label=f'Sample {index}')

    ax_amp.set_ylabel('Amplitude')
    ax_amp.grid(True)
    ax_pha.set_ylabel('Phase')
    ax_pha.set_xlabel('Subcarrier')
    ax_pha.grid(True)

    plt.legend()
    #plt.subtitle(f'{filename}')
    plt.tight_layout()
    plt.show()


def print_csi_info(samples, index):
    if index >= samples.nsamples:
        print(f"Index {index} is out of bounds. Total samples: {samples.nsamples}")
        return

    csi = samples.get_csi(index, config.remove_null_subcarriers, config.remove_pilot_subcarriers)
    if csi is None:
        print(f"No CSI data available for sample {index}.")
        return

    amp = np.abs(csi)
    phase = np.angle(csi)

    print(f"Sample {index}:")
    print(f"Amplitude: {amp}")
    print(f"Phase: {phase}")



def handle_commands(samples, bandwidth, filename, valid_packets_indices):
    while True:
        command = input('> ')

        if 'help' in command:
            print(config.help_str)
        elif 'exit' in command:
            break
        elif '-' in command and all(string_is_int(part) for part in command.split('-')):
            start, end = map(int, command.split('-'))
            plot_csi_range(samples, start, end, bandwidth, filename, valid_packets_indices)
        elif string_is_int(command):
            index = int(command)
            print_csi_info(samples, index)
            plot_csi(samples, index, bandwidth, filename, valid_packets_indices)
        else:
            print('Unknown command. Type help.')

def process_pcap_file(filepath):
    try:
        samples = decoder.read_pcap(filepath)
        print(f"Successfully read {len(samples.csi)} samples.")
        valid_packets_indices = []

        for index in range(samples.nsamples):
            csi = samples.get_csi(index, config.remove_null_subcarriers, config.remove_pilot_subcarriers)
            if csi is None:
                continue

            nsub = int(samples.bandwidth * 3.2)
            x = np.arange(-1 * nsub / 2, nsub / 2)

            if validate_csi_packet(csi, x, index) is None:
                continue
            else:
                valid_packets_indices.append(index)

        print(len(valid_packets_indices))
        print(f"Valid packets indices: {valid_packets_indices}")
        return samples, valid_packets_indices
    except FileNotFoundError:
        print(f'File {filepath} not found.')
        exit(-1)
    except Exception as e:
        print(f"An error occurred: {e}")
        exit(-1)

if __name__ == "__main__":
    pcap_filename = input('Pcap file name: ')
    if '.pcap' not in pcap_filename:
        pcap_filename += '.pcap'
    pcap_filepath = '/'.join([config.pcap_fileroot, pcap_filename])

    samples, valid_packets_indices = process_pcap_file(pcap_filepath)
    if samples:
        handle_commands(samples, samples.bandwidth, pcap_filename, valid_packets_indices)
    else:
        print("No valid CSI samples to process.")

