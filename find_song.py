import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy import signal
import argparse
from collections import namedtuple, defaultdict, Counter
from operator import itemgetter
import pickle


# <nfft>/2 frequency bins are split logarithmically into <nbins> frequency bands
def generate_bands(nbands, nfft):
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
    freq_bins = generate_bands(no_bands, nfft)
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
def generate_fingerprints(peaks):
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
    for i in range(0, len(peaks) - zone_size, 1):
        target_zones.append(peaks[i:i + zone_size])

    # Now we generate "addresses" using anchor point for each target zone,
    # address tuple format will be: (<anchor_freq>, <point_freq>, <dt>)
    # where dt is delta time between anchor and point
    # We can generate fingerprint with anchor point which is 3 points before
    # start point of each TZ
    for t in range(delay, len(target_zones) - 1):
        target_zone = target_zones[t]
        anchor = peaks[t-delay]
        anchor_time = anchor[0]
        anchor_freq = anchor[1]
        for point in target_zone:
            point_time = point[0]
            point_freq = point[1]
            address = (anchor_freq, point_freq, point_time - anchor_time)
            fingerprints.append((address, anchor_time))

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
    # Hash table of record is key-value type (<address> -> <anchor_time>)
    # Database is hash table generated for number of songs that can then be recognized
    for address in record:
        # If record address (key) is in database, then it's possible that is our song
        if address in database:
            # We can have same address for multiple songs,
            # append each (song ID) one to possible match
            for song in database[address]:
                for song_metadata in song:
                    possible.append(song_metadata)

    # Check for number of common target zones
    zone_size = 5
    common = {}
    for match in possible:
        if match not in common.keys():
            common[match] = 0
        common[match] += 1

    filtered_possible = [key[0] for key, val in common.items() if val >= zone_size]

    # TODO: needs improvement for time coherency
    cnt = Counter(filtered_possible)
    tmp_solution = cnt.most_common()
    print(cnt.most_common(3))

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

    nfft = 1024
    # Needed only for spectrogram data, not really interested in plotting now
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