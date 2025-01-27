import os
import numpy as np
import importlib
from sklearn.model_selection import train_test_split
import config
from process import lowpass, hampel_filter
from sklearn.preprocessing import MinMaxScaler

decoder = importlib.import_module(f'decoders.{config.decoder}')

def read_pcap(file_path):
    csi_data = []
    samples = decoder.read_pcap(file_path)
    for index in range(samples.nsamples):
        csi = samples.get_csi(index, config.remove_null_subcarriers, config.remove_pilot_subcarriers)
        if csi is None or not np.any(csi):
            print(f"Empty CSI data in file: {file_path}")
            continue
        csi_data.append(csi)
    if len(csi_data) == 0:
        print(f"No valid CSI data in file: {file_path}")
    return csi_data


def process_csi_data(csi_packets):
    # 删除特定的索引
    processed_data = []
    scaler = MinMaxScaler()
    for csi in csi_packets:
        cutoff = 20
        fs = 50  # sample
        '''USE FILTER
        csi_lowpass_filtered = lowpass(csi, cutoff, fs, order=5)
        csi_abs = np.absolute(csi_lowpass_filtered)
        # csi_abs_final = hampel_filter(csi_abs_pre, 5, 3)
        csi_phase = np.angle(csi_lowpass_filtered)
    
        '''
        csi_lowpass_filtered = lowpass(csi, cutoff, fs, order=5)
        csi_abs = np.absolute(csi_lowpass_filtered)
        csi_phase = np.angle(csi_lowpass_filtered)
        csi_abs = np.absolute(csi)
        csi_phase = np.angle(csi)
        csi_combined = np.vstack((csi_abs, csi_phase)).T
        #csi_combined = np.concatenate((csi_abs, csi_phase))
        csi_normalized = scaler.fit_transform(csi_combined)
        processed_data.append(csi_normalized)
    return np.stack(processed_data)

#[300,468]
def process_csi_data1(csi_packets):
    processed_data = []
    scaler = MinMaxScaler()
    for csi in csi_packets:
        cutoff = 20
        fs = 50
        csi_lowpass_filtered = lowpass(csi, cutoff, fs, order=5)
        csi_abs = np.absolute(csi_lowpass_filtered)
        csi_phase = np.angle(csi_lowpass_filtered)

        # Combine amplitude and phase
        csi_combined = np.concatenate((csi_abs, csi_phase), axis=-1)

        # 归一化处理
        csi_normalized = scaler.fit_transform(csi_combined.reshape(-1, csi_combined.shape[-1]))
        csi_normalized = csi_normalized.reshape(csi_combined.shape)
        processed_data.append(csi_normalized)

    return np.stack(processed_data)


def process_csi_data2(csi):

    cutoff = 20
    fs = 100
    csi_lowpass_filtered = lowpass(csi, cutoff, fs, order=5)
    csi_abs = np.absolute(csi_lowpass_filtered)
    # csi_abs_final = hampel_filter(csi_abs_pre, 5, 3)
    csi_phase = np.angle(csi_lowpass_filtered)
    csi_tensor = np.hstack((csi_abs, csi_phase))
    return csi_tensor


# This function is used to load and process all gesture data
def load_data(gesture_data_path):
    gestures = ['circle','down','left','push','stretch']
    data = [] # CSI
    labels = []

    for i, gesture in enumerate(gestures):
        gesture_path = os.path.join(gesture_data_path, gesture)
        # Iterate through each date subdirectory in the gestures folder
        for date_dir in os.listdir(gesture_path):
            date_dir_path = os.path.join(gesture_path, date_dir)
            if os.path.isdir(date_dir_path):
                for root, dirs, files in os.walk(date_dir_path):
                    for pcap_file in files:
                        if pcap_file.endswith('.pcap'):
                            file_path = os.path.join(root, pcap_file)
                            csi_packets = read_pcap(file_path)
                            processed_csi = process_csi_data(csi_packets)  #
                            data.append(processed_csi)
                            labels.append(i)

    return np.array(data), np.array(labels)

def load_data_specfic(file_path):
    if not file_path.endswith('.pcap'):
        raise ValueError("MUST BE .pcap")

    # Read and process individual PCAP files
    csi_packets = read_pcap(file_path)
    processed_csi = process_csi_data(csi_packets)


    print("After process:", processed_csi.shape)


    return processed_csi


#test
def load_data_test(gesture_data_path):
    gestures = ['circle','down','left','push','stretch']
    data = []
    labels = []

    for i, gesture in enumerate(gestures):
        gesture_path = os.path.join(gesture_data_path, gesture)  #
        for pcap_file in os.listdir(gesture_path):
            if pcap_file.endswith('.pcap'):
                file_path = os.path.join(gesture_path, pcap_file)
                csi_packets = read_pcap(file_path)  # return data
                processed_csi = process_csi_data(csi_packets)
                data.append(processed_csi)
                labels.append(i)

    return np.array(data), np.array(labels)


gesture_data_path = 'C:\\Users\\ludal\\PycharmProjects\\pythonProject\\data'
#print(gesture_data_path)
# load
data, labels = load_data(gesture_data_path)
print(f"Data shape: {data.shape}")
#print(f"Labels shape: {labels.shape}")
