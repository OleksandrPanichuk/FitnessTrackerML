# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction
from FrequencyAbstraction import FourierTransformation
from sklearn.cluster import KMeans

# %%
df = pd.read_pickle("../../data/interim/02_outliers_removed_chauvenet.pkl")
df.info()


# %%

predictor_columns = list(df.columns[:6])

# %%
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20, 5)
plt.rcParams["figure.dpi"] = 100
plt.rcParams["lines.linewidth"] = 2

# %%

subset = df[df["set"] == 35]["gyr_y"].plot()
# %%


for col in predictor_columns:
    df[col].interpolate(inplace=True)

df.info()
# %%
df[df["set"] == 25]["acc_y"].plot()
# %%
df[df["set"] == 50]["acc_y"].plot()

# %%

duration = df[df["set"] == 1].index[-1] - df[df["set"] == 1].index[0]
# %%

duration.seconds

# %%

for s in df["set"].unique():
    start = df[df["set"] == s].index[0]
    stop = df[df["set"] == s].index[-1]

    duration = stop - start

    df.loc[df["set"] == s, "duration"] = duration.seconds
# %%
df.head()
# %%
duration_df = df.groupby(["category"])["duration"].mean()
# %%
duration_df.iloc[0] / 5
# %%
duration_df.iloc[1] / 10


# %%
df_lowpass = df.copy()

LowPass = LowPassFilter()

fs = 1000 / 200  # 5 instances per second
cutoff = 1.3

df_lowpass = LowPass.low_pass_filter(df_lowpass, "acc_y", fs, cutoff, order=5)
df_lowpass.head()

# %%


subset = df_lowpass[df_lowpass["set"] == 45]

fix, ax = plt.subplots(nrows=2, sharex=True, figsize=(20, 10))
ax[0].plot(subset["acc_y"].reset_index(drop=True), label="Raw data")
ax[1].plot(
    subset["acc_y_lowpass"].reset_index(drop=True), label="Low-pass filtered data"
)
ax[0].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)
ax[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), fancybox=True, shadow=True)
# %%

for col in predictor_columns:
    df_lowpasss = LowPass.low_pass_filter(df_lowpass, col, fs, cutoff, order=5)
    df_lowpass[col] = df_lowpasss[col + "_lowpass"]
    del df_lowpasss[col + "_lowpass"]
# %%
df_pca = df_lowpass.copy()
PCA = PrincipalComponentAnalysis()

# %%
pca_values = PCA.determine_pc_explained_variance(df_pca, predictor_columns)
# %%

plt.figure(figsize=(10, 10))
plt.plot(range(1, len(pca_values) + 1), pca_values)
plt.xlabel("Number of Principal Components")
plt.ylabel("Explained Variance (%)")
plt.show()
# %%
df_pca = PCA.apply_pca(df_pca, predictor_columns, number_comp=3)
df_pca.head()
# %%

subset = df_pca[df_pca["set"] == 35]
subset[["pca_1", "pca_2", "pca_3"]].reset_index(drop=True).plot()
# %%

df_squared = df_pca.copy()

acc_r = df_squared["acc_x"] ** 2 + df_squared["acc_y"] ** 2 + df_squared["acc_z"] ** 2
gyr_r = df_squared["gyr_x"] ** 2 + df_squared["gyr_y"] ** 2 + df_squared["gyr_z"] ** 2

df_squared["acc_r"] = np.sqrt(acc_r)
df_squared["gyr_r"] = np.sqrt(gyr_r)
# %%
df_squared.head()
# %%
subset = df_squared[df_squared["set"] == 12]
subset[["acc_r", "gyr_r"]].reset_index(drop=True).plot(subplots=True)

# %%

df_temporal = df_squared.copy()
NumAbs = NumericalAbstraction()

# %%

predictor_columns += ["acc_r", "gyr_r"]

window_size = int(1000 / 200)

# %%
df_temporal = NumAbs.abstract_numerical(
    df_temporal, predictor_columns, window_size, "mean"
)
df_temporal = NumAbs.abstract_numerical(
    df_temporal, predictor_columns, window_size, "std"
)

df_temporal.head(20)
# %%


df_temporal_list = []

for s in df_temporal["set"].unique():
    subset = df_temporal[df_temporal["set"] == s].copy()

    subset = NumAbs.abstract_numerical(subset, predictor_columns, window_size, "mean")
    subset = NumAbs.abstract_numerical(subset, predictor_columns, window_size, "std")

    df_temporal_list.append(subset)

# %%
df_temporal = pd.concat(df_temporal_list)
# %%
df_temporal.info()
# %%
subset[["acc_y", "acc_y_temp_mean_ws_5", "acc_y_temp_std_ws_5"]].plot()
# %%
subset[["gyr_y", "gyr_y_temp_mean_ws_5", "gyr_y_temp_std_ws_5"]].plot()

# %%
df_freq = df_temporal.copy().reset_index()
# %%
FreqAbs = FourierTransformation()
# %%

fs = int(1000 / 200)
ws = int(2800 / 200)

# %%

df_freq = FreqAbs.abstract_frequency(df_freq, ["acc_y"], ws, fs)

# %%
df_freq.columns
# %%

subset = df_freq[df_freq["set"] == 1]
# %%
subset[["acc_y"]].plot()
# %%
subset[
    [
        "acc_y_max_freq",
        "acc_y_freq_weighted",
        "acc_y_pse",
        "acc_y_freq_1.429_Hz_ws_14",
        "acc_y_freq_2.5_Hz_ws_14",
    ]
].plot()

# %%


df_freq_list = []

for s in df_freq["set"].unique():
    subset = df_freq[df_freq["set"] == s].reset_index(drop=True).copy()
    subset = FreqAbs.abstract_frequency(subset, predictor_columns, ws, fs)
    df_freq_list.append(subset)

# %%
df_freq = pd.concat(df_freq_list).set_index("epoch (ms)", drop=True)

# %% Dealing with overlapping

df_freq = df_freq.dropna()
df_freq = df_freq.iloc[::2]

# %% Clustering

df_cluster = df_freq.copy()

cluster_columns = ["acc_x", "acc_y", "acc_z"]
k_values = range(2, 10)
inertias = []


for k in k_values:
    subset = df_cluster[cluster_columns]

    kmeans = KMeans(n_clusters=k, n_init=20, random_state=0)
    cluster_labels = kmeans.fit_predict(subset)

    inertias.append(kmeans.inertia_)


# %%

plt.figure(figsize=(10, 10))
plt.plot(k_values, inertias)
plt.xlabel("Number of clusters (k)")
plt.ylabel("Inertia")
plt.show()

# %%

kmeans = KMeans(n_clusters=5, n_init=20, random_state=0)
subset = df_cluster[cluster_columns]
df_cluster["cluster"] = kmeans.fit_predict(subset)

# %%
df_cluster
# %% Plot clusters

fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")


for c in df_cluster["cluster"].unique():
    subset = df_cluster[df_cluster["cluster"] == c]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=c)
    
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.legend()
plt.show()

# %%

fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(projection="3d")


for l in df_cluster["label"].unique():
    subset = df_cluster[df_cluster["label"] == l]
    ax.scatter(subset["acc_x"], subset["acc_y"], subset["acc_z"], label=l)
    
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.set_zlabel("Z-axis")
plt.legend()
plt.show()
# %% Export final dataset


df_cluster.to_pickle("../../data/interim/03_data_features.pkl")
# %%
