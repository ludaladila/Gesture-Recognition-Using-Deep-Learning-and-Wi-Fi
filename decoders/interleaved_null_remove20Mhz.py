'''
Interleaved
===========

Fast and efficient methods to extract
Interleaved CSI samples in PCAP files.

~230k samples per second.

Suitable for bcm43455c0 and bcm4339 chips.

Requires Numpy.

Usage
-----

import decoders.interleaved as decoder

samples = decoder.read_pcap('path_to_pcap_file')

Bandwidth is inferred from the pcap file, but
can also be explicitly set:
samples = decoder.read_pcap('path_to_pcap_file', bandwidth=40)
'''

__all__ = [
    'read_pcap'
]

import os
import numpy as np

# Indexes of Null and Pilot OFDM subcarriers
# https://www.oreilly.com/library/view/80211ac-a-survival/9781449357702/ch02.html
nulls = {

    20: [x + 32 for x in [
        -32, -31, -30, -29,
        31, 30, 29, 28, 0
    ]],

    40: [x + 64 for x in [
        -64, -63, -62, -61, -60, -59, -1,
        63, 62, 61, 60, 59, 1, 0
    ]],

    80: [x + 128 for x in [
        -128, -127, -126, -125, -124, -123, -1,
        127, 126, 125, 124, 123, 1, 0
    ]],

    160: [x + 256 for x in [
        -256, -255, -254, -253, -252, -251, -129, -128, -127, -5, -4, -3, -2, -1,
        255, 254, 253, 252, 251, 129, 128, 127, 5, 4, 3, 3, 1, 0
    ]]
}

'''nulls = {

    20: [x + 32 for x in [
        -32, -31, -30, -29,
        31, 30, 29,0
    ]],

    40: [x + 64 for x in [
        -64, -63, -62, -61, -60, -59, -1,
        63, 62, 61, 60, 59, 1, 0
    ]],

    80: [x + 128 for x in [
        -128, -127, -126, -125, -124, -123, -1,
        127, 126, 125, 124, 123, 1, 0
    ]],

    160: [x + 256 for x in [
        -256, -255, -254, -253, -252, -251, -129, -128, -127, -5, -4, -3, -2, -1,
        255, 254, 253, 252, 251, 129, 128, 127, 5, 4, 3, 3, 1, 0
    ]]
}
'''
pilots = {
    20: [x + 32 for x in [
        -21, -7,
        21, 7
    ]],

    40: [x + 64 for x in [
        -53, -25, -11,
        53, 25, 11
    ]],

    80: [x + 128 for x in [
        -103, -75, -39, -11,
        103, 75, 39, 11
    ]],

    160: [x + 256 for x in [
        -231, -203, -167, -139, -117, -89, -53, -25,
        231, 203, 167, 139, 117, 89, 53, 25
    ]]
}


class SampleSet(object):
    '''
        A helper class to contain data read
        from pcap files.
    '''

    def __init__(self, samples, bandwidth, nsamples_max):
        self.rssi, self.fctl, self.mac, self.seq, self.css, self.csi = samples
        self.nsamples = self.csi.shape[0]

        self.bandwidth = bandwidth
        self.nsamples_max = nsamples_max

    def get_rssi(self, index):
        return self.rssi[index]

    def get_fctl(self, index):
        return self.fctl[index]

    def get_mac(self, index):
        return self.mac[index * 6: (index + 1) * 6]


    def get_seq(self, index):
        sc = int.from_bytes(  # uint16: SC
            self.seq[index * 2: (index + 1) * 2],
            byteorder='little',
            signed=False
        )
        fn = sc % 16  # Fragment Number
        sc = int((sc - fn) / 16)  # Sequence Number

        return (sc, fn)

    def get_css(self, index):
        return self.css[index * 2: (index + 1) * 2]

    def get_csi(self, index, rm_nulls=False, rm_pilots=False):
        csi = self.csi[index].copy()
        if rm_nulls or rm_pilots:
            valid_indices = list(range(csi.shape[0]))
            if rm_nulls:
                valid_indices = [idx for idx in valid_indices if idx not in nulls[self.bandwidth]]
            if rm_pilots:
                valid_indices = [idx for idx in valid_indices if idx not in pilots[self.bandwidth]]

            csi = csi[valid_indices]

        return csi


    def remove_null_pilots(self, index, rm_nulls=False, rm_pilots=False):
        csi = self.csi[index].copy()
        if rm_nulls or rm_pilots:
            valid_indices = list(range(csi.shape[0]))
            if rm_nulls:
                valid_indices = [idx for idx in valid_indices if idx not in nulls[self.bandwidth]]
            if rm_pilots:
                valid_indices = [idx for idx in valid_indices if idx not in pilots[self.bandwidth]]

            csi = csi[valid_indices]

        return csi




    def print(self, index):
        # Mac ID
        macid = self.get_mac(index).hex()
        macid = ':'.join([macid[i:i + 2] for i in range(0, len(macid), 2)])

        # Sequence control
        sc, fn = self.get_seq(index)

        # Core and Spatial Stream
        css = self.get_css(index).hex()

        rssi = self.get_rssi(index)
        fctl = self.get_fctl(index)

        print(
            f'''
Sample #{index}
---------------
Source Mac ID: {macid}
Sequence: {sc}.{fn}
Core and Spatial Stream: 0x{css}
RSSI: {rssi}
FCTL: {fctl}
            '''
        )

    def get_valid_samples_indices(self):
        valid_samples_indices = []
        for index in range(self.nsamples):
            csi = self.get_csi(index, rm_nulls=False, rm_pilots=False)
            nsub = int(self.bandwidth * 3.2)
            x = np.arange(-1 * nsub / 2, nsub / 2)

            # Iterate over the range of CSI values
            for i in range(len(x) - 9):
                # Check that the amplitude of 10 consecutive frequencies are all less than 50
                if np.all(np.abs(csi[i:i + 10]) < 50):
                    break
            else:
                valid_samples_indices.append(index)

        return valid_samples_indices

    def create_subset(self, valid_indices):
        # Convert bytearray to numpy array to support list indexing
        rssi_subset = np.array(self.rssi)[valid_indices]
        fctl_subset = np.array(self.fctl)[valid_indices]
        mac_subset = np.array(self.mac).reshape(-1, 6)[valid_indices, :].flatten()
        seq_subset = np.array(self.seq).reshape(-1, 2)[valid_indices, :].flatten()
        css_subset = np.array(self.css).reshape(-1, 2)[valid_indices, :].flatten()
        csi_subset = self.csi[valid_indices, :]

        subset = SampleSet(
            (
                rssi_subset,
                fctl_subset,
                mac_subset,
                seq_subset,
                css_subset,
                csi_subset,
            ),
            self.bandwidth,
            len(valid_indices)
        )
        return subset

    def valid_samples(self):
        valid_indices = self.get_valid_samples_indices()
        if len(valid_indices) < 100:
            print("Warning: Fewer than 100 valid samples available.")
        # If there are less than 100 valid samples, use all valid samples
        valid_indices = valid_indices[:100]
        return self.create_subset(valid_indices)


def __find_bandwidth(incl_len):
    '''
        Determines bandwidth
        from length of packets.

        incl_len is the 4 bytes
        indicating the length of the
        packet in packet header
        https://wiki.wireshark.org/Development/LibpcapFileFormat/

        This function is immune to small
        changes in packet lengths.
    '''
    # Convert byte sequences to integers
    pkt_len = int.from_bytes(
        incl_len,
        byteorder='little',
        signed=False
    )
    # The number of bytes before we
    # have CSI data is 60. By adding
    # 128-60 to frame_len, bandwidth
    # will be calculated correctly even
    # if frame_len changes +/- 128
    # Some packets have zero padding.
    # 128 = 20 * 3.2 * 4
    nbytes_before_csi = 60
    pkt_len += (128 - nbytes_before_csi)

    bandwidth = 20 * int(
        pkt_len // (20 * 3.2 * 4)
    )

    return bandwidth


def __find_nsamples_max(pcap_filesize, nsub):
    '''
        Returns an estimate for the maximum possible number
        of samples in the pcap file.

        The size of the pcap file is divided by the size of
        a packet to calculate the number of samples. However,
        some packets have a padding of a few bytes, so the value
        returned is slightly higher than the actual number of
        samples in the pcap file.
    '''

    # PCAP global header is 24 bytes
    # PCAP packet header is 12 bytes
    # Ethernet + IP + UDP headers are 46 bytes
    # Nexmon metadata is 18 bytes
    # CSI is nsub*4 bytes long
    #
    # So each packet is 12 + 46 + 18 + nsub*4 bytes long
    nsamples_max = int(
        (pcap_filesize - 24) / (
                12 + 46 + 18 + (nsub * 4)
        )
    )

    return nsamples_max


def read_pcap(pcap_filepath, bandwidth=0, nsamples_max=0, rm_nulls=False, rm_pilots=False):
    '''
        Reads CSI samples from
        a pcap file. A SampleSet
        object is returned.

        Bandwidth and maximum samples
        are inferred from the pcap file by
        default, but you can also set them explicitly.
    '''

    pcap_filesize = os.stat(pcap_filepath).st_size
    with open(pcap_filepath, 'rb') as pcapfile:
        fc = pcapfile.read()

    if bandwidth == 0:
        bandwidth = __find_bandwidth(
            # 32-36 is where the incl_len
            # bytes for the first frame are
            # located.
            # https://wiki.wireshark.org/Development/LibpcapFileFormat/
            fc[32:36]
        )
    # Number of OFDM sub-carriers
    nsub = int(bandwidth * 3.2)

    if nsamples_max == 0:
        nsamples_max = __find_nsamples_max(pcap_filesize, nsub)

    # Preallocating memory
    rssi = bytearray(nsamples_max * 1)
    fctl = bytearray(nsamples_max * 1)
    mac = bytearray(nsamples_max * 6)
    seq = bytearray(nsamples_max * 2)
    css = bytearray(nsamples_max * 2)
    csi = bytearray(nsamples_max * nsub * 4)

    # Pointer to current location in file.
    # This is faster than using file.tell()
    # =24 to skip pcap global header
    ptr = 24

    nsamples = 0
    while ptr < pcap_filesize:
        # Read frame header
        # Skip over Eth, IP, UDP
        ptr += 8
        frame_len = int.from_bytes(
            fc[ptr: ptr + 4],
            byteorder='little',
            signed=False
        )
        ptr += 50

        # 2 bytes: Magic Bytes               @ 0 - 1
        # 1 bytes: RSSI                      @ 2 - 2
        # 1 bytes: FCTL                      @ 3 - 3
        # 6 bytes: Source Mac ID             @ 4 - 10
        # 2 bytes: Sequence Number           @ 10 - 12
        # 2 bytes: Core and Spatial Stream   @ 12 - 14
        # 2 bytes: ChanSpec                  @ 14 - 16
        # 2 bytes: Chip Version              @ 16 - 18
        # nsub*4 bytes: CSI Data             @ 18 - 18 + nsub*4

        rssi[nsamples] = fc[ptr + 2]
        fctl[nsamples] = fc[ptr + 3]
        mac[nsamples * 6: (nsamples + 1) * 6] = fc[ptr + 4: ptr + 10]
        seq[nsamples * 2: (nsamples + 1) * 2] = fc[ptr + 10: ptr + 12]
        css[nsamples * 2: (nsamples + 1) * 2] = fc[ptr + 12: ptr + 14]
        csi[nsamples * (nsub * 4): (nsamples + 1) * (nsub * 4)] = fc[ptr + 18: ptr + 18 + nsub * 4]

        ptr += (frame_len - 42)
        nsamples += 1

    # Convert CSI bytes to numpy array
    csi_np = np.frombuffer(
        csi,
        dtype=np.int16,
        count=nsub * 2 * nsamples
    )
    if rm_nulls or rm_pilots:
        valid_indices = list(set(range(nsub)) - set(nulls[bandwidth]) - set(pilots[bandwidth]))
        csi_np = csi_np[:, np.r_[valid_indices, valid_indices] + np.array([0, nsub])]

    # Cast numpy 1-d array to matrix
    csi_np = csi_np.reshape((nsamples, nsub * 2))

    # Convert csi into complex numbers
    csi_cmplx = np.fft.fftshift(
        csi_np[:nsamples, ::2] + 1.j * csi_np[:nsamples, 1::2], axes=(1,)
    )

    # Convert RSSI to Two's complement form
    rssi = np.frombuffer(rssi, dtype=np.int8, count=nsamples)

    sample_set = SampleSet((rssi, fctl, mac, seq, css, csi_cmplx), bandwidth, nsamples_max)
    return sample_set.valid_samples()  # Returning only the valid samples


if __name__ == "__main__":
    samples = read_pcap('circle17.pcap')
    print(f'Max_Sample: {samples.nsamples_max}')
    print(f'Sample: {samples.nsamples}')
    #print(samples.csi)
    csi_first_sample = samples.get_csi(0, rm_nulls=True, rm_pilots=True)
    print(csi_first_sample)
    print(len(csi_first_sample))

    valid_indices = samples.get_valid_samples_indices()
    print(valid_indices)
    valid_samples = samples.create_subset(valid_indices)

    print(f'Sample: {valid_samples.nsamples}')
