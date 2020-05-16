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


def get_peaks_skimage(data):
    peaks = peak_local_max(data, min_distance=50)
    return peaks


# <nfft>/2 frequency bins are split logarithmically into <nbins> frequency bands
def generate_bins(nbins, nfft):
    bands = [0]
    for i in range(int(np.log2(nfft)) - nbins, int(np.log2(nfft))):
        bands.append(2**i)
    print("freq bands:", bands)
    return bands


def prune_peaks(peaks, freq_bins, time_size=50):
    # Number of time slices for each segment to filter
    curr_time = time_size
    pruned_peaks = []
    tmp_peaks = defaultdict(list)
    ampl_coeff = 1.5
    for peak in peaks:
        # After we iterate over <time_size> slices, find mean and filter
        if peak.time > curr_time:
            for binned_peaks in tmp_peaks.values():
                if len(binned_peaks) != 0:
                    mean_amplitude = np.mean([p.amplitude for p in binned_peaks])
                    pruned_peaks += list(filter(lambda p: p.amplitude > mean_amplitude*ampl_coeff, binned_peaks))
                    #pruned_peaks += [x for x in binned_peaks if x.amplitude >= mean_amplitude*ampl_coeff]
            tmp_peaks.clear()
            curr_time += time_size
        # If still on current segment, add segment peaks
        tmp_peaks[np.searchsorted(freq_bins, peak.freq, side='right')].append(peak)
    print("no. of prunned peaks:", len(pruned_peaks))

    return pruned_peaks


def find_peaks(spec_data, nfft):
    Peak = namedtuple('Peak', ['amplitude', 'freq', 'time'])
    peaks = []
    time_slice = 0
    no_bins = 6
    freq_bins = generate_bins(no_bins, nfft)
    # Shape returns (n,m) where n - number of rows, m - number of columns
    # In <specData>, frequency bins increase down the rows, time slices increase across the columns
    # Get each time slice (column) and its neighbours
    for i in range(1, spec_data.shape[1] - 1):
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
    # Now we have filtered spectrogram points that can be further
    # optimized by taking number of time slices (instead of one at the time)
    # and calculating its average amplitude and then applying same method as above
    print("no. of peaks:", len(peaks))
    pruned_peaks = prune_peaks(peaks, freq_bins)

    # Amplitudes are not needed anymore, therefore we can omit them
    # and we will sort points by increasing time and then frequency

    filtered_peaks = [(p.time, p.freq) for p in pruned_peaks]
    filtered_peaks.sort(key=itemgetter(0, 1))

    return filtered_peaks


# <peaks> is now list of (time, freq) tuples
def generate_fingerprints(peaks, song_id):
    # More points <zone_size> in each target zone could
    # increase efficiency but also increase search time
    zone_size = 5
    target_zones = []
    # Each target zone overlaps with previous in zoneSize - 1 points
    for i in range(0, len(peaks) - zone_size, 1):
        target_zones.append(peaks[i:i+zone_size])

    # Now we generate "addresses" using anchor point for each target zone,
    # address tuple format will be: (<anchor_freq>, <point_freq>, <dt>)
    # where dt is delta time between anchor and point
    # We skip first <delay> target zones (TZ) because their anchor could
    # not be located properly, but there should be enough fingerprints
    # even without it
    delay = 3
    fingerprints = []
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
    # Now we can generate fingerprint with anchor point which is 3 points before
    # start point of each TZ
    for t in range(delay, len(target_zones) - 1):
        target_zone = target_zones[t]
        anchor = peaks[t-delay]
        anchor_time = anchor[0]
        anchor_freq = anchor[1]
        for p in target_zone:
            p_time = p[0]
            p_freq = p[1]
            address = (anchor_freq, p_freq, p_time-anchor_time)
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
    for song in glob.glob(os.path.join(file_path, '*.wav')):
        print(song)
        # Read .wav file and convert it to mono channel
        fs, signal_data = wavfile.read(song)
        if signal_data.ndim == 2:
            signal_data = np.mean(signal_data, axis=1)

        # Decimation - apply AA filter and the downsample by factor q
        signal_data = signal.decimate(signal_data, q=4, ftype='fir')

        nfft = 256
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
        fingerprints = generate_fingerprints(peaks, song_id=song)
        hash_table = generate_hashtable(fingerprints)

        with open('database.sz', 'wb') as file:
            pickle.dump(hash_table, file)
        id += 1
        print("Done")

    return


if __name__ == "__main__":
    main()