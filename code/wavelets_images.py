# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 10:59:24 2021

@author: Joshua
"""

import pywt
import matplotlib.pyplot as plt
import seaborn as sns

# dwt wavelets: "db4", "sym4", "coif4", "haar", "dmey"
[phi, psi, x] = pywt.Wavelet('db4').wavefun(level=10)
fig, ax =plt.subplots(1,2)
sns.lineplot(x=x, y=phi, ax=ax[0])
sns.lineplot(x=x, y=psi, ax=ax[1])
ax[0].set(xlabel='x', ylabel='',title="Scaling Function")
ax[1].set(xlabel='x', ylabel='',title="Wavelet Function")
fig.suptitle("Example approximations of a Daubechies-4 wavelet")
plt.tight_layout()
fig.show()
plt.savefig("../masters_thesis/images/example_db4_wavelet.pdf")

[phi, psi, x] = pywt.Wavelet('sym4').wavefun(level=10)
fig, ax =plt.subplots(1,2)
sns.lineplot(x=x, y=phi, ax=ax[0])
sns.lineplot(x=x, y=psi, ax=ax[1])
ax[0].set(xlabel='x', ylabel='',title="Scaling Function")
ax[1].set(xlabel='x', ylabel='',title="Wavelet Function")
fig.suptitle("Example approximations of a Symlets-4 wavelet")
plt.tight_layout()
fig.show()
plt.savefig("../masters_thesis/images/example_sym4_wavelet.pdf")

[phi, psi, x] = pywt.Wavelet('coif4').wavefun(level=10)
fig, ax =plt.subplots(1,2)
sns.lineplot(x=x, y=phi, ax=ax[0])
sns.lineplot(x=x, y=psi, ax=ax[1])
ax[0].set(xlabel='x', ylabel='',title="Scaling Function")
ax[1].set(xlabel='x', ylabel='',title="Wavelet Function")
fig.suptitle("Example approximations of a Coiflets-4 wavelet")
plt.tight_layout()
fig.show()
plt.savefig("../masters_thesis/images/example_coif4_wavelet.pdf")

[phi, psi, x] = pywt.Wavelet('haar').wavefun(level=10)
fig, ax =plt.subplots(1,2)
sns.lineplot(x=x, y=phi, ax=ax[0])
sns.lineplot(x=x, y=psi, ax=ax[1])
ax[0].set(xlabel='x', ylabel='',title="Scaling Function")
ax[1].set(xlabel='x', ylabel='',title="Wavelet Function")
fig.suptitle("Example approximations of a Haar wavelet")
plt.tight_layout()
fig.show()
plt.savefig("../masters_thesis/images/example_haar_wavelet.pdf")


[phi, psi, x] = pywt.Wavelet('dmey').wavefun(level=10)
fig, ax =plt.subplots(1,2)
sns.lineplot(x=x, y=phi, ax=ax[0])
sns.lineplot(x=x, y=psi, ax=ax[1])
ax[0].set(xlabel='x', ylabel='',title="Scaling Function")
ax[1].set(xlabel='x', ylabel='',title="Wavelet Function")
fig.suptitle("Example approximations of a D.-Mey. wavelet")
plt.tight_layout()
fig.show()
plt.savefig("../masters_thesis/images/example_dmey_wavelet.pdf")