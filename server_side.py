import datetime
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tick
from scipy.io import wavfile
from scipy import signal
import argparse
from skimage.feature import peak_local_max
from collections import namedtuple, defaultdict, Counter
from operator import itemgetter
import os
import glob
import pickle
import json
import io


def get_peaks_skimage(data):
    peaks = peak_local_max(data, min_distance=50)
    return peaks


# <nfft>/2 frequency bins are split logarithmically into <nbins> frequency bands
def generate_bins(nbins, nfft):
    # TODO change these hardcoded bands, currently suitable for nfft = 1024
    bands = [0, 10, 20, 40, 80, 160, 511]
    print("freq bands:", bands)
    return bands


def prune_peaks(peaks, freq_bins, time_size=50):
    # Number of time slices for each segment to filter
    curr_time_segment = time_size
    pruned_peaks = []
    tmp_peaks = []
    ampl_coef = 1
    for peak in peaks:
        # After we iterate over <time_size> slices, find mean and filter
        if peak.time > curr_time_segment:
            mean_amplitude = np.mean([p.amplitude for p in tmp_peaks])
            [pruned_peaks.append(p) for p in tmp_peaks if p.amplitude >= mean_amplitude]
            tmp_peaks.clear()
            curr_time_segment += time_size
        # If still on current segment, add segment peaks
        tmp_peaks.append(peak)
    print("no. of pruned peaks:", len(pruned_peaks))

    return pruned_peaks


def find_peaks(spec_data, nfft):
    Peak = namedtuple('Peak', ['amplitude', 'freq', 'time'])
    peaks = []
    no_bands = 6
    freq_bins = generate_bins(no_bands, nfft)
    # Shape returns (n,m) where n - number of rows, m - number of columns
    # In <spec_data>, frequency bins increase down the rows, time slices increase across the columns
    # Get each time slice (column) and its neighbours
    for time_slice in range(1, spec_data.shape[1] - 1):
        #tmp_peaks = defaultdict(list)
        tmp_peaks = []
        time_slice_mean = 0

        for j in range(0, no_bands):
            curr_band = [freq_bins[j], freq_bins[j+1]]
            curr_fft = spec_data[curr_band[0]:curr_band[1], time_slice]
            # get index of strongest bin in current frequency band
            index = np.argmax(curr_fft)
            time_slice_mean += curr_fft[index]
            peak = Peak(amplitude=curr_fft[index], freq=curr_band[0]+index, time=time_slice)
            #tmp_peaks[np.searchsorted(freq_bins, peak.freq, side='right')].append(peak)
            tmp_peaks.append(peak)

        # Keep only bins above time slice mean
        time_slice_mean /= no_bands
        [peaks.append(p) for p in tmp_peaks if p.amplitude >= time_slice_mean]


    '''
        fft_prev = spec_data[:, i-1]
        fft = spec_data[:, i]
        fft_next = spec_data[:, i+1]
        # Iterate over each frequency bin:
        # in order to be classified as local maxima,
        # fft needs to be larger than its four neighbours
        tmp_peaks = defaultdict(list)
        for j in range(1, freq_bins[-1]):
            if(fft[j] > fft[j-1] and
               fft[j] > fft[j+1] and
               fft[j] > fft_prev[j] and
               fft[j] > fft_next[j]):
                peak = Peak(amplitude=fft[j], freq=j, time=time_slice)
                # Create key-value pair, in which key is one of nbins freq bins
                # and value is an array of Peak namedtuples
                tmp_peaks[np.searchsorted(freq_bins, peak.freq, side='right')].append(peak)
        # Find max value out of possible (local maximas)
        # for each of created <nbins> (logarithmic) freq bins
        bin_peaks = [max(x, key=lambda p: p.amplitude) for x in tmp_peaks.values()]
        # Out of this <nbins> or less powerful amplitudes,
        # find the mean value and keep only bins above this mean value
        if len(bin_peaks) == 0:
            continue
        mean_amplitude = np.mean([p.amplitude for p in bin_peaks])
        amplitude_coeff = 1.5
        for p in bin_peaks:
            # TODO: find optimal value for amplCoefficient
            # Some parts of the song are very quiet, e.g. beginning or ending,
            # which will cause low mean value compared to rest of the song and
            # false 'high amplitude' frequencies, <amplitude_coeff> can help with that
            if p.amplitude >= mean_amplitude*amplitude_coeff:
                peaks.append(p)
        tmp_peaks.clear()
        bin_peaks.clear()
        time_slice += 1 
    '''

    # Now we have filtered spectrogram points that can be further
    # optimized by taking number of time slices (instead of one at the time)
    # and calculating its average amplitude and then applying same method as above
    print("no. of peaks:", len(peaks))
    pruned_peaks = prune_peaks(peaks, freq_bins)

    # Amplitudes are not needed anymore, therefore we can omit them
    # and we will sort points by increasing time and then frequency
    filtered_peaks = [(p.time, p.freq) for p in pruned_peaks]
    filtered_peaks.sort(key=itemgetter(0, 1))
    #print(filtered_peaks)

    return filtered_peaks


# <peaks> is now list of (time, freq) tuples
def generate_fingerprints(peaks, song_id):
    # More points <zone_size> in each target zone could
    # increase accuracy but also increase search time
    zone_size = 5
    target_zones = []
    delay = 3

    # We skip first <delay> target zones (TZ) because their anchor could
    # not be located properly, but there should be enough fingerprints
    # even without it
    fingerprints = []
    # Each target zone overlaps with previous in zone_size - 1 points
    for i in range(delay, len(peaks) - zone_size, 1):
        target_zones.append(peaks[i:i+zone_size])

    '''
    anchor_time = peaks[0][0]
    anchor_freq = peaks[0][1]
    for t in range(0, delay):
        target_zone = target_zones[t]
        for p in target_zone:
            p_time = p[0]
            p_freq = p[1]
            address = (anchor_freq, p_freq, p_time-anchor_time)
            fingerprints.append((address, (song_id, anchor_time)))
    '''
    # Now we generate "addresses" using anchor point for each target zone,
    # address tuple format will be: (<anchor_freq>, <point_freq>, <dt>)
    # where dt is delta time between anchor and point
    # We can generate fingerprint with anchor point which is 3 points before
    # start point of each TZ
    for t in range(0, len(target_zones) - 1):
        target_zone = target_zones[t]
        anchor = peaks[t]
        anchor_time = anchor[0]
        anchor_freq = anchor[1]
        for point in target_zone:
            point_time = point[0]
            point_freq = point[1]
            address = (anchor_freq, point_freq, point_time-anchor_time)
            fingerprints.append((address, (song_id, anchor_time)))

    # return list of tuples (<address>, (<song_id>, <anchor_time>))
    return fingerprints


def generate_hashtable(fingerprints):
    hashtable = {}
    # Hash address - tuple(<anchor_freq>, <point_freq>, <dt>)
    # Hash value - tuple(<song_id>, <anchor_time>)
    for fingerprint in fingerprints:
        hash_address = fingerprint[0]
        hash_value = fingerprint[1]
        if hash_address not in hashtable:
            hashtable[hash_address] = []
        hashtable[hash_address].append(hash_value)

    return hashtable


def main():
    # Initiate the parser
    text = "Spectrogram module allows for computing and plotting spectrogram from .wav file"
    parser = argparse.ArgumentParser(description=text)
    parser.add_argument("-p", "--plot", help="Plot spectrogram of given .wav file")
    parser.add_argument("-d", "--data", help="Compute spectrogram data of given .wav file")
    parser.add_argument("-c", "--create", help="Create database of recognizable songs")
    args = parser.parse_args()
    plot_spec = False
    if args.plot:
        print("Plotting spectrogram of %s" % args.plot)
        file_path = args.plot
        plot_spec = True
    elif args.data:
        print("Creating spectrogram data for %s" % args.data)
        file_path = args.data
    elif args.create:
        print("Creating database of songs...")
        file_path = args.create
    else:
        print("No .wav file specified")
        return

    # TODO add error checking for file type and missing files
    id = 0
    all_hash_tables = []
    for song in glob.glob(os.path.join(file_path, '*.wav')):
        print(song)
        # Read .wav file and convert it to mono channel
        fs, signal_data = wavfile.read(song)
        if signal_data.ndim == 2:
            signal_data = np.mean(signal_data, axis=1)

        # Decimation - apply AA filter and the downsample by factor q
        signal_data = signal.decimate(signal_data, q=4, ftype='fir')

        nfft = 1024
        # Frequency resolution of spectrogram is given by formula (fs / q) / nfft,
        # defaults to (44100 / 4) / 1024 ~ 10.8 Hz
        # Plots Pxx = 10*log10(abs(Sxx)), where Sxx = Zxx**2 (Zxx returned by signal.stft)
        # Power Spectral Density plot
        pxx, f, t, im = plt.specgram(signal_data,
                                     NFFT=nfft,
                                     Fs=fs,
                                     window=signal.hann(nfft),
                                     noverlap=nfft/2,
                                     scale_by_freq=False,
                                     cmap=plt.get_cmap('viridis'))

        if plot_spec:
            plt.title("Spectrogram of %s" % song)
            plt.xlabel("Time [s]")
            plt.ylabel("Frequency [Hz]")
            plt.show()

        peaks = find_peaks(pxx, nfft)
        fingerprints = generate_fingerprints(peaks, song_id=id)
        hash_table = generate_hashtable(fingerprints)
        all_hash_tables.append(hash_table)
        id += 1

    # Merge all hash tables into one and make database file from it
    super_dict = defaultdict(list)
    for ht in all_hash_tables:
        for k, v in ht.items():
            super_dict[k].append(v)

    with open('database.sz', 'wb') as file:
        pickle.dump(super_dict, file)

    return


if __name__ == "__main__":
    main()