# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 11:52:27 2021

@author: Joshua
"""

import numpy as np
from scipy import fftpack
from scipy import signal
from matplotlib import pyplot as plt
import pywt
from scipy.interpolate import griddata

np.random.seed(42)


# Frequency and sampling rate
f1 = 10 # frequency
f2 = 50
Fs = 1000 # sampling rate
t = np.arange(0,1,1/1000)
# Sine function
y1 = np.sin(2 * np.pi * f1 * t)
y2 = np.sin(2 * np.pi * f2 * t)
y2[:500] = 0
y1[-500:] = 0


# both signals
fig, (ax1, ax2) = plt.subplots(2, sharex='col',
                        gridspec_kw={'hspace': 0})
fig.suptitle('Sine waves at 10 Hz (upper) and 50 Hz (lower)')
ax1.plot(t,y1)
ax2.plot(t,y2)
#ax1.axis(xmin=0,xmax=1)
ax1.set(ylabel="Signal Amplitude")
ax2.set(ylabel='Signal Amplitude', xlabel="Time[s]")
fig.show()
fig.savefig("../masters_thesis/images/sine_waves_10_50")


y = y1+y2
plt.figure(figsize=(6, 5))
plt.plot(y, label='Original signal')

# Perform Fourier transform using scipy
y_fft = fftpack.fft(y)

# Plot data
n = np.size(t)
fr = Fs/2 * np.linspace(0,1,int(n/2))
y_m = 2/n * abs(y_fft[0:np.size(fr)])

fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
ax[0].plot(t, y)    # plot time series
ax[1].plot(fr, y_m) # plot freq domain
fig.show()

# mixed signal and fft power
fig, (ax1,ax2) = plt.subplots(1,2,gridspec_kw={'wspace': 0.4})
ax1.plot(t, y)
ax1.set(ylabel='Signal Amplitude',xlabel='Time [s]')

ax2.plot(fr, y_m)
ax2.set_xlim(0,80)
ax2.set(ylabel='Frequency Magnitude',xlabel='Frequency [Hz]')
fig.show()
fig.savefig("../masters_thesis/images/ts_freq_example")


# stft plots at different resolutions
fs1, t1, Zxx1 = signal.stft(y, Fs, nperseg=100)
fst, tt, Zxxt = signal.stft(y, Fs, nperseg=500)
fs2, t2, Zxx2 = signal.stft(y, Fs, nperseg=1000)

fig, axs = plt.subplots(1,3,sharey="row",
                        gridspec_kw={'wspace': 0.1})
im1 = axs[0].pcolormesh(t1, fs1,np.abs(Zxx1), vmin=0, vmax=1, shading='auto',cmap="jet")
axs[0].axis(ymin=0,ymax=100)
axs[0].axis(xmin=0,xmax=1)
axs[0].set(ylabel='Frequency [Hz]', xlabel="Time[s]", title="Size=0.1s")
im2 = axs[1].pcolormesh(tt, fst, np.abs(Zxxt), vmin=0, vmax=1, shading='auto',cmap="jet")
axs[1].set(xlabel="Time[s]", title="Size=0.5s")
im3 = axs[2].pcolormesh(t2, fs2, np.abs(Zxx2), vmin=0, vmax=1, shading='auto',cmap="jet")
axs[2].yaxis.tick_right()
axs[2].set(xlabel="Time[s]", title="Size=1s")
fig.colorbar(im1, ax=axs[2])
fig.show()
fig.savefig("../masters_thesis/images/stft_example")



# cwt example, lower frequency plots for cwt analysis
scales = np.arange(1,100,1)
waveletname = 'morl'
coeff, freq = pywt.cwt(y, scales, waveletname,1)

# create scalogram

fig, (ax1,ax2) = plt.subplots(2, sharex='col',
                        gridspec_kw={'hspace': 0})
ax1.pcolormesh(t, scales,np.abs(coeff), shading="auto",cmap="jet")
ax1.set(ylabel='Scale')
ax1.hlines(16,xmin = 0,xmax=1, color="red")
ax1.hlines(81,xmin = 0,xmax=1, color="red")
ax1.set_xlim(left=0,right=1)
ax2.plot(t,abs(coeff[0,:]))
ax2.set(ylabel='First CWT coeff',xlabel='Time [s]')
fig.show()
fig.savefig("../masters_thesis/images/cwt_example")

# test_f is a mapping from scales to pseudo-frequencies
test_f = pywt.scale2frequency(waveletname, scales)/ (1/Fs)


# dwt example
