import os
import pandas as pd
import numpy as np
from scipy.signal import lfilter, butter
from tqdm import tqdm


def envelope_spectrum(x, fs, speed, order=5):
    """
    get envelope spectrum

    :param x: input acceleration
    :param fs: samplerate (Hz)
    :param order: order of the butterworth bandpass filter
    :return:
    """

    def interpolate_spectrum(f_rep_order, rep_spectrum_order):
        f_rep_hz_new = np.linspace(0, 30, num=1000)
        rep_spectrum_hz_new = np.interp(f_rep_hz_new, f_rep_order, rep_spectrum_order)
        rep_spectrum_hz_new = np.nan_to_num(rep_spectrum_hz_new)
        return f_rep_hz_new,  rep_spectrum_hz_new

    nyq = 0.5 * fs  # nyquist frequency
    x = x - np.mean(x)

    b, a = butter(order, [500 / nyq, 4000 / nyq], btype='band')
    x_env = np.abs(lfilter(b, a, x))
    x_env = (x_env - np.mean(x_env))/np.std(x_env)
    x_env_spectrum = np.abs(np.fft.rfft(x_env))/np.sqrt(len(x_env))
    f_env = np.fft.rfftfreq(len(x_env))*fs
    return interpolate_spectrum(f_env/(speed/60), x_env_spectrum)



#%%
load0 = {0: "/97.mat", 1: "/105.mat", 2: "/169.mat", 3: "/209.mat", 4: "/118.mat", 5: "/185.mat", 6: "/222.mat", 7: "/130.mat", 8: "/197.mat", 9: "/234.mat"}
load1 = {0: "/98.mat", 1: "/106.mat", 2: "/170.mat", 3: "/210.mat", 4: "/119.mat", 5: "/186.mat", 6: "/223.mat", 7: "/131.mat", 8: "/198.mat", 9: "/235.mat"}
load2 = {0: "/99.mat", 1: "/107.mat", 2: "/171.mat", 3: "/211.mat", 4: "/120.mat", 5: "/187.mat", 6: "/224.mat", 7: "/132.mat", 8: "/199.mat", 9: "/236.mat"}
load3 = {0: "/100.mat", 1: "/108.mat", 2: "/172.mat", 3: "/212.mat", 4: "/121.mat", 5: "/188.mat", 6: "/225.mat", 7: "/133.mat", 8: "/200.mat", 9: "/237.mat"}

load = [load0, load1, load2, load3]

length = 4096
sample = 100


#%%
print("Preprocessing for actual data")
df = pd.read_parquet('cwru.parquet', engine='pyarrow')

healthyX = {}
healthyY = {}
data_load = []
label_load = []

for i, l in enumerate(load):
    for j in tqdm(range(0,10)):
        txt = l[j]
        df_load_per_mat = df[df["url"].str.contains(txt) & df["sensor location"].str.contains("DE")]
        assert len(df_load_per_mat) == 1
        series = df_load_per_mat.iloc[0]
        acc = series["acceleration"]
        samplerate = series["samplerate"]

        if not np.isnan(series["test rpm"]):
            speed = series["test rpm"]
        else:
            speed = series["Approx. Motor Speed (rpm)"]
        
        acc_chunks = [acc[k:k + length] for k in range(0, len(acc) - length + 1 , (len(acc) - length)//(sample - 1) )]
        acc_chunks = acc_chunks[:sample]
        acc_chunks = [envelope_spectrum(x, samplerate, speed)[1] for x in acc_chunks]
        data_load.extend(acc_chunks)
        label_load.extend([j]*len(acc_chunks))
        if j == 0:
            healthyX[i] = data_load[:]
            healthyY[i] = label_load[:]

data_load, label_load = np.array(data_load), np.array(label_load)
np.save("XreallDEenv.npy", data_load)
np.save("yreallDEenv.npy", label_load)

#%%
print("Preprocessing for synthetic data")
df = pd.read_parquet('cwru_synthetic.parquet', engine='pyarrow')

healthyidx = (label_load == 0)
healthyX = data_load[healthyidx]
healthyY = label_load[healthyidx]
data_load = []
label_load = []

for i, l in enumerate(load):
    for j in tqdm(range(1,10)):
        txt = l[j]
        df_load_per_mat = df[df["url"].str.contains(txt) & df["sensor location"].str.contains("DE")]
        assert len(df_load_per_mat) == 1
        series = df_load_per_mat.iloc[0]
        acc = series["acceleration"]
        samplerate = series["samplerate"]

        if not np.isnan(series["test rpm"]):
            speed = series["test rpm"]
        else:
            speed = series["Approx. Motor Speed (rpm)"]
        
        acc_chunks = [acc[k:k + length] for k in range(0, len(acc) - length + 1 , (len(acc) - length)//(sample - 1) )]
        acc_chunks = acc_chunks[:sample]
        acc_chunks = [envelope_spectrum(x, samplerate, speed)[1] for x in acc_chunks]
        data_load.extend(acc_chunks)
        label_load.extend([j]*len(acc_chunks))

data_load.extend(healthyX)
label_load.extend(healthyY)
data_load, label_load = np.array(data_load), np.array(label_load)

np.save("XsynallDEenv.npy", data_load)
np.save("ysynallDEenv.npy", label_load)
