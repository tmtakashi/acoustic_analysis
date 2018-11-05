import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sounddevice as sd

from acoustic_analysis import tsp_sychronous_addition, sychronous_addition, thd
from tsp_design import pink_tsp
from sinewave_generator import sinewave

def tsp_measurement(title, device_number):
    fs = 48000
    n = 18
    N = 2 ** n
    tsp, tsp_inv = pink_tsp(18, gain=80, repeat=4)

    sd.default.device = device_number
    recording = sd.playrec(tsp, fs, channels=1)
    sd.wait()

    recording = np.squeeze(recording)
    np.savez(title + '_tsp.npz', recording=recording)

    recording_mean = tsp_sychronous_addition(recording, 4, 2**18)
    
    H = np.fft.fft(recording_mean) * np.fft.fft(tsp_inv)
    f = np.linspace(0, fs/2, len(H), endpoint=False)

    plt.plot(f, 20*np.log10(np.abs(H) / np.max(np.abs(H))))
    plt.title(title)
    plt.xlabel('Frequency[Hz]')
    plt.ylabel('Level[dB]')
    plt.xlim(20, 20000)
    plt.xscale('log')

    plt.savefig(title + '.png', dpi=300)
    

def thd_measurement(speaker_name, device_number):
    
    fs = 48000
    duration = 3
    frequencies = [125, 250, 500, 1000, 2000, 4000, 5000]

    recordings = np.zeros((len(frequencies), fs * duration))
    thd_list = np.zeros(len(frequencies))
    for i, frequency in enumerate(frequencies):
        signal = sinewave(frequency, duration, fs)
        sd.default.device = device_number
        recording = sd.playrec(signal, fs, channels=1)
        sd.wait()

        recording = np.squeeze(recording)
        recording_mean = sychronous_addition(recording, frequency, 5)
        thd_list[i] = thd(recording_mean, frequency, 4)

        recordings[i, :] = recording

    np.savez(speaker_name + '_thd.npz', recordings=recordings)
    thd_series = pd.Series(thd_list, index=[str(f) + '[Hz]' for f in frequencies])
    thd_series.to_csv(speaker_name + '.csv')
    print(thd_series)

tsp_measurement('No23', [2, 2])
thd_measurement('No23', [2, 2])




