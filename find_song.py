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

# <nfft>/2 frequency bins are split logarithmically into <nbins> frequency bands
def generate_bins(nbins, nfft):
    bands = [0]
    for i in range(int(np.log2(nfft)) - nbins, int(np.log2(nfft))):
        bands.append(2**i)
    print("freq bands:", bands)
    return bands


def prune_peaks(peaks, freq_bins, time_size=500):
    # Number of time slices for each segment to filter
    curr_time = time_size
    prunned_peaks = []
    tmp_peaks = defaultdict(list)
    ampl_coef = 1.5
    for peak in peaks:
        # After we iterate over <time_size> slices, find mean and filter
        if peak.time > curr_time:
            for binned_peaks in tmp_peaks.values():
                if len(binned_peaks) != 0:
                    mean_ampl = np.mean([p.amplitude for p in binned_peaks])
                    prunned_peaks += list(filter(lambda p: p.amplitude > mean_ampl * ampl_coef, binned_peaks))
                    #prunned_peaks += [x for x in binned_peaks if x.amplitude >= mean_ampl*ampl_coef]
            tmp_peaks.clear()
            curr_time += time_size
        # If still on current segment, add segment peaks
        tmp_peaks[np.searchsorted(freq_bins, peak.freq, side='right')].append(peak)
    print("no. of prunned peaks:", len(prunned_peaks))

    return prunned_peaks


def find_peaks(spec_data, nfft):
    nbins = 6
    peaks = []
    time_slice = 0
    Peak = namedtuple('Peak', ['amplitude', 'freq', 'time'])
    freq_bins = generate_bins(nbins, nfft)
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
        mean_ampl = np.mean([p.amplitude for p in bin_peaks])
        ampl_coef = 1.2
        for p in bin_peaks:
            # TODO: find optimal value for amplCoefficient
            # Some parts of the song are very quiet, e.g. beginning or ending,
            # which will cause low mean value compared to rest of the song and
            # false 'high amplitude' frequencies, ampl_coef can help with that
            if p.amplitude >= mean_ampl*ampl_coef:
                peaks.append(p)
        tmp_peaks.clear()
        bin_peaks.clear()
        time_slice += 1
    # Now we have filtered spectrogram points that can be further
    # optimized by taking number of time slices (instead of one at the time)
    # and calculating its average amplitude and then applying same method as above
    print("no. of peaks:", len(peaks))
    prunned_peaks = prune_peaks(peaks, freq_bins)

    # Amplitudes are not needed anymore, therefore we can omit them
    # and we will sort points by increasing time and then frequency
    '''
    filtered = defaultdict(list)
    for x in prunnedPeaks:
        bisect.insort(filtered[x.time], x.freq)
    '''
    filtered_peaks = [(p.time, p.freq) for p in prunned_peaks]
    filtered_peaks.sort(key=itemgetter(0, 1))

    return filtered_peaks


# <peaks> is now list of (time, freq) tuples
def generate_fingerprints(peaks):
    # More points <zone_size> in each target zone could
    # increase efficiency but also increase search time
    zone_size = 5
    target_zones = []
    # Each target zone overlaps with previous in zoneSize - 1 points
    for i in range(0, len(peaks) - zone_size, 1):
        target_zones.append(peaks[i:i+zone_size])
    #print(target_zones)
    # Now we generate "addresses" using anchor point for each target zone,
    # address tuple format will be: (<anchor_freq>, <point_freq>, <dt>)
    # where dt is delta time between anchor and point
    # Although not correct, we take first point as an anchor for
    # first <delay> target zones
    '''
    anchor_time = peaks[0][0]
    anchor_freq = peaks[0][1]
    for t in range(0, delay):
        target_zone = target_zones[t]
        for p in target_zone:
            p_time = p[0]
            p_freq = p[1]
            address = (anchor_freq, p_freq, p_time-anchor_time)
            fingerprints.append((address, anchor_time))
    '''
    # Now we can generate with anchor point which is 3 points before
    # start point of each target zone
    delay = 3
    fingerprints = []
    for t in range(delay, len(target_zones) - 1):
        target_zone = target_zones[t]
        anchor = peaks[t-delay]
        anchor_time = anchor[0]
        anchor_freq = anchor[1]
        for p in target_zone:
            p_time = p[0]
            p_freq = p[1]
            address = (anchor_freq, p_freq, p_time-anchor_time)
            fingerprints.append((address, anchor_time))

    #print(fingerprints)
    # return list of tuples (<address>, <anchor_time>)
    return fingerprints


def generate_hashtable(fingerprints):
    hashtable = {}
    # Hash address - tuple(<anchor_freq>, <point_freq>, <dt>)
    # Hash value - <anchor_time>
    for fingerprint in fingerprints:
        hash_address = fingerprint[0]
        hash_value = fingerprint[1]
        if hash_address not in hashtable:
            hashtable[hash_address] = []
        hashtable[hash_address].append(hash_value)

    return hashtable


def find_song(record, database):
    possible = []
    couples = defaultdict(list)
    # Record hash table is key-value type (<address> -> <anchor_time>)
    # Database is hash table generated for number of songs that can then be recognized
    for address in record.keys():
        # If record address (key) is in database, then it's possible that is our song
        if address in database:
            for song_metadata in database[address]:
                possible.append(song_metadata[0])
    tmp_solution = Counter(possible).most_common()

    '''
    found = {}
    for match in possible:
        anchor_freq = match[0]
        for song_metadata in match[1]:
            if (anchor_freq, song_metadata) not in found.keys():
                found[(anchor_freq, song_metadata)] = 0
            found[(anchor_freq, song_metadata)] += 1
    '''
    return tmp_solution


def main():
    # Initiate the parser
    text = "find_song module tries to find song in database given input .wav file"
    parser = argparse.ArgumentParser(description=text)
    parser.add_argument("-f", "--find", help="Find song info")
    args = parser.parse_args()

    fs, signal_data = wavfile.read(args.find)
    if signal_data.ndim == 2:
        signal_data = np.mean(signal_data, axis=1)

    # Decimation - apply AA filter and the downsample by factor q
    signal_data = signal.decimate(signal_data, q=4, ftype='fir')

    nfft = 256
    # Needed only for Pxx, not really interested in plotting now
    pxx, f, t, im = plt.specgram(signal_data,
                                 NFFT=nfft,
                                 Fs=fs,
                                 window=signal.hann(nfft),
                                 noverlap=nfft//2,
                                 cmap=plt.get_cmap('viridis'))

    peaks = find_peaks(pxx, nfft)
    fingerprints = generate_fingerprints(peaks)
    song_hash_table = generate_hashtable(fingerprints)

    # Open database of known songs
    with open('database.sz', 'rb') as db:
        database = pickle.load(db)
    print("Database loaded")
    print("Searching for matches...")
    results = find_song(song_hash_table, database)
    if len(results) == 0:
        print("Could not recognize song")
    else:
        best_match = results[0][0]
        print("Best match ===> ", best_match)

    return


if __name__ == "__main__":
    main()