import numpy as np
from scipy.signal import firwin
import soundfile as sf

def nextpow2(n):
    '''
    Returns the next exponent of base2.

    n: input number (int)
    '''
    m_f = np.log2(n)
    m_i = np.ceil(m_f)
    return int(np.log2(2**m_i))

def reference_value(filename):
    '''
    Returns value of the reference.
    Calibration level = 94 dBSPL

    filename : Name of wav file (str)
    '''
    data, fs = sf.read(filename)
    calib_e = np.max(data) / np.sqrt(2) #RMS of input signal
    reference =  10**(-4.7) * calib_e
    return reference

def thd(data, f0, order):
    '''
    Returns total harmonic distortion of a signal.

    filename : Name of wav file (str)
    f0 : Fundamental frequency (float)
    order : Order of the overtone (int)
    '''
    # data, fs = sf.read(filename)
    # if data.ndim == 2: data = data[:, 0]
    
    numtaps = 2**nextpow2(len(data))

    data = np.r_[data, np.zeros(numtaps - len(data))] * np.hamming(numtaps)
    
    fs = 48000
    powers = np.zeros(order)
    for i in range(order):
        # 1/3 oct bandwidth
        f1 = (i + 1) * (f0 / 2**(1/6)) / (fs/2)
        f2 = (i + 1) * (f0 * 2**(1/6)) / (fs/2)
        bpf = firwin(numtaps, [f1, f2], pass_zero=False)
        filtered_spectrum = np.fft.fft(data, numtaps) * np.fft.fft(bpf, numtaps)
        powers[i] = np.max(np.abs(filtered_spectrum))
    
    thd = np.sqrt(np.sum((powers[1:order]**2))) / powers[0] 

    return thd

def thdn(filename, f0):
    '''
    Prints total harmonic distortion of a signal.

    filename : Name of wav file (str)
    f0 : Fundamental frequency (float)
    '''
    data, fs = sf.read(filename)
    if data.ndim == 2: data = data[:, 0]
    
    numtaps = 2**nextpow2(len(data))

    data = np.r_[data, np.zeros(numtaps - len(data))] * np.hanning(numtaps)
    data_spectrum = np.fft.fft(data, numtaps)
    # 1/3 oct bandwidth
    f1 = (f0 / 2**(1/6)) / (fs/2)
    f2 = (f0 * 2**(1/6)) / (fs/2)
    bpf = firwin(numtaps, [f1, f2], pass_zero=False)
    filtered_spectrum = data_spectrum * np.fft.fft(bpf, numtaps)
    left_over_spectrum = np.fft.fft(data, numtaps)

    thdn = np.sqrt(np.sum(np.abs(left_over_spectrum))) / np.max(np.abs(filtered_spectrum))

    print(str(f0) + "Hz: " + str(np.round(100*thdn, decimals=2)) + '%, ' + str(np.round(20*np.log10(thdn/100), decimals=2)) + "[dB]")

def plot_directivity(speaker_output_file, pulse_file):
    '''
    Plots directivity using output from the speaker and output pulse from the turn table.

    speaker_output_file : Name of wav file of speaker output(str)
    pulse_file : Name of wav file of turn table output(str)
    '''
    speaker_output, fs = sf.read(speaker_output_file)
    pulse, fs = sf.read(pulse_file)

    pulse_index = np.where(pulse >= np.max(pulse) - 0.01*np.max(pulse))[0]
    turn_period = pulse_index[1] - pulse_index[0]

    angle = np.arange(360)


def tsp_sychronous_addition(data, repeat, N):
    '''
    Returns sychronous addtion of an audio data.

    filename : Name of wav file (str)
    repeat : Number of repetition (int)
    N : Length of input signal (int)
    '''
    # data, fs = sf.read(filename)

    # if data.ndim == 2: data = data[:, 0]

    # add zeros if length is too short
    if len(data) < repeat * N:
        data = np.r_[data, np.zeros(repeat * N - len(data))]

    mean = np.zeros(N)
    for i in range(repeat + 1):
        print(data[i * N : (i + 1) * N].shape)
        mean = mean + data[i * N : (i + 1) * N]
    mean = mean / repeat

    return mean

def sychronous_addition(data, frequency, num_period):
    # data, fs = sf.read(filename)

    # if data.ndim == 2: data = data[:, 0]
    fs = 48000
    # tap number of a frame
    N = num_period * fs // frequency
    # numbers of frames
    frame_num = len(data) // N 

    mean = np.zeros(N)
    for i in range(frame_num):
        mean += data[i * N : (i + 1) * N]
    mean /= frame_num

    return mean 