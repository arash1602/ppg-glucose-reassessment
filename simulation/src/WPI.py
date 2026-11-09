import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dA_dCg_df = pd.read_csv("results/mean_metrics/Metric_Mean_dA_dCg_total.csv", index_col = "wavelength_nm")
beta_derm_df = pd.read_csv("results/mean_metrics/Metric_Mean_beta_derm.csv", index_col = "wavelength_nm")
power_dB_df = pd.read_csv("results/mean_metrics/Metric_Mean_power_dB.csv", index_col = "wavelength_nm")



def normalize(df):
    minimum = df.min().min()
    maximum = df.max().max()


    return (df - minimum) / (maximum - minimum)



glucose_score = normalize(dA_dCg_df)
dermal_score = normalize(beta_derm_df)
power_score = normalize(power_dB_df)


fig, ax = plt.subplots()
glucose_score.plot(ax = ax, label = "Glucose Score", color = "red")
dermal_score.plot(ax = ax, label = "Dermal Sensitivity Score", color = "green")
power_score.plot(ax = ax, label = "Power Score", color = "black")

# Calculate Wavelength Performance Index
WPI = (
    0.33 * glucose_score
    + 0.33 * dermal_score
    + 0.33 * power_score
)



# Plot WPI
WPI.plot(figsize=(10, 6))

plt.xlabel("Wavelength (nm)")
plt.ylabel("WPI")
plt.title("Wavelength Performance Index")
plt.grid(True)
plt.show()