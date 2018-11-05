import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from acoustic_analysis import thd

fs = 48000
t = 3

n = np.arange(0, t * fs)

x = np.cos(2*np.pi*1000*n/fs)+8e-4*np.sin(2*np.pi*2000*n/fs)+2e-5*np.cos(2*np.pi*3000*n/fs-np.pi/4) + 8e-6*np.sin(2*np.pi*4000*n/fs)

# spectrum = np.abs((np.fft.fft(x) / len(n)) * 2) 
# freq = np.fft.fftfreq(len(n), 1/fs)
# plt.plot(freq,spectrum)
# plt.xlim(0, 5000)
# plt.show()

sf.write('thd_check.wav', x, fs)

thd('thd_check.wav', 1000, 5)

print(np.sqrt((0.5/2) * 4) * np.sqrt(2))