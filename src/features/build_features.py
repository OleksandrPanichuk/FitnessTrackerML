# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from DataTransformation import LowPassFilter, PrincipalComponentAnalysis
from TemporalAbstraction import NumericalAbstraction



# %%
df = pd.read_pickle("../../data/interim/02_outliers_removed_chauvenet.pkl")
df.info()


# %%

predictor_columns = list(df.columns[:6])

# %%
plt.style.use("fivethirtyeight")
plt.rcParams["figure.figsize"] = (20,5)
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

duration = df[df["set"] == 1].index[-1] -  df[df["set"] == 1].index[0]
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
duration_df= df.groupby(["category"])["duration"].mean()
# %%
duration_df.iloc[0] / 5
# %%
duration_df.iloc[1] / 10




# %%
df_lowpass = df.copy()

LowPass = LowPassFilter()

fs = 1000 / 200 # 5 instances per second
cutoff = 1.3

df_lowpass = LowPass.low_pass_filter(df_lowpass, "acc_y", fs, cutoff, order=5)
df_lowpass.head()

# %%


subset = df_lowpass[df_lowpass["set"] == 45]

fix, ax = plt.subplots(nrows=2, sharex=True, figsize=(20,10))
ax[0].plot(subset["acc_y"].reset_index(drop=True), label="Raw data")
ax[1].plot(subset["acc_y_lowpass"].reset_index(drop=True), label="Low-pass filtered data")
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

plt.figure(figsize=(10,10))
plt.plot(range(1, len(pca_values)+1), pca_values)
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
subset[["acc_r","gyr_r"]].reset_index(drop=True).plot(subplots=True)