import os
import sys
import argparse
import glob
import subprocess

parser = argparse.ArgumentParser(description="Convert .mp3 files to .wav format")
parser.add_argument('inputDirectory',
                    help='Path to the input directory.')

songs_dir = parser.parse_args(sys.argv[1:])

if os.path.exists(songs_dir.inputDirectory):
    for song in glob.glob(os.path.join(songs_dir.inputDirectory, '*.mp3')):
        base = os.path.splitext(os.path.basename(song))[0]
        dst = songs_dir.inputDirectory + base + '.wav'
        subprocess.call(['ffmpeg', '-i', song,
                        dst])
        os.remove(song)

else:
    print("Given path doesn't exist!")
