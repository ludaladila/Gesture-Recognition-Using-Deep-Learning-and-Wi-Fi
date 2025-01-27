import importlib
import numpy as np
import matplotlib.pyplot as plt
import config_visual

decoder = importlib.import_module(f'decoders.{config_visual.decoder}')

def string_is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def validate_csi_packet(csi, x):
    idx_65 = np.where(x == -65)[0][0]
    idx_100 = np.where(x == 100)[0][0]
    if np.abs(csi[idx_65]) < 200 or np.abs(csi[idx_100]) < 200:
        return False
    return True



def plot_csi(samples, index, bandwidth, filename, valid_packets_indices):
    if index not in valid_packets_indices:
        print(f"Sample {index} is invalid and will not be plotted.")
        return

    csi = samples.get_csi(index, config_visual.remove_null_subcarriers, config_visual.remove_pilot_subcarriers)
    if csi is None:
        print(f"No CSI data available for sample {index}.")
        return

    nsub = int(bandwidth * 3.2)
    x = np.arange(-1 * nsub / 2, nsub / 2)

    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(x, np.abs(csi), label=f'Amplitude Index {index}')
    plt.ylabel('Amplitude')
    plt.grid(True)
    #plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(x, np.angle(csi, deg=True), label=f'Phase Index {index}')
    plt.ylabel('Phase')
    plt.xlabel('Subcarrier')
    plt.suptitle(f'{filename}')
    plt.grid(True)
    #plt.legend()
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

        csi = samples.get_csi(index, config_visual.remove_null_subcarriers, config_visual.remove_pilot_subcarriers)
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
    plt.tight_layout()
    plt.show()

def print_csi_info(samples, index, valid_packets_indices):
    if index not in valid_packets_indices:
        print(f"Sample {index} is invalid.")
        return

    csi = samples.get_csi(index, config_visual.remove_null_subcarriers, config_visual.remove_pilot_subcarriers)
    if csi is None:
        print(f"No CSI data available for sample {index}.")
        return

    amp = np.abs(csi)
    phase = np.angle(csi)

    print(f"Sample {index}:")
    print(f"Amplitude: {amp}")
    print(f"Phase: {phase}")
1-5
def handle_commands(samples, bandwidth, filename, valid_packets_indices):
    while True:
        command = input('> ')

        if 'help' in command:
            print(config_visual.help_str)
        elif 'exit' in command:
            break
        elif '-' in command and all(string_is_int(part) for part in command.split('-')):
            start, end = map(int, command.split('-'))
            plot_csi_range(samples, start, end, bandwidth, filename, valid_packets_indices)
        elif string_is_int(command):
            index = int(command)
            if index in valid_packets_indices:
                samples.print(index)
                print_csi_info(samples, index, valid_packets_indices)
                plot_csi(samples, index, bandwidth, filename, valid_packets_indices)
            else:
                print(f"Sample {index} is invalid.")
        else:
            print('Unknown command. Type help.')

def process_pcap_file(filepath):
    try:
        samples = decoder.read_pcap(filepath)
        print(f"Successfully read {len(samples.csi)} samples.")
        valid_packets_indices = []

        for index in range(samples.nsamples):
            csi = samples.get_csi(index, config_visual.remove_null_subcarriers, config_visual.remove_pilot_subcarriers)
            if csi is None:
                continue
            nsub = int(samples.bandwidth * 3.2)
            x = np.arange(-1 * nsub / 2, nsub / 2)

            if validate_csi_packet(csi, x):
                valid_packets_indices.append(index)

        print(f"Valid packets indices: {valid_packets_indices},Total count: {len(valid_packets_indices)}")
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
    pcap_filepath = '/'.join([config_visual.pcap_fileroot, pcap_filename])

    samples, valid_packets_indices = process_pcap_file(pcap_filepath)
    if samples:
        handle_commands(samples, samples.bandwidth, pcap_filename, valid_packets_indices)
    else:
        print("No valid CSI samples to process.")
