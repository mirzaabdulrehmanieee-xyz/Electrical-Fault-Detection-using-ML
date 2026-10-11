
import pandas as pd
import matplotlib.pyplot as plt
import os

# Create the figures folder if it does not exist
os.makedirs("results/figures", exist_ok=True)

df = pd.read_csv("data/detect_dataset.csv")

df = df.drop(columns=["Unnamed: 7", "Unnamed: 8"])
df = df.rename(columns={"Output (S)": "fault"})

# Take the first 2000 rows
data = df.iloc[:2000]

# Plot the three currents
plt.figure(figsize=(12, 5))

plt.plot(data.index, data["Ia"], label="Ia")
plt.plot(data.index, data["Ib"], label="Ib")
plt.plot(data.index, data["Ic"], label="Ic")

# Shade the regions where a fault occurs
fault_rows = data["fault"] == 1
plt.fill_between(
    data.index, 0, 1,
    where=fault_rows,
    transform=plt.gca().get_xaxis_transform(),
    alpha=0.2,
    color="red",
    label="Fault"
)

plt.xlabel("Row number")
plt.ylabel("Current (A)")
plt.title("Three-Phase Currents for First 2000 Rows")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the first figure
plt.savefig(
    "results/figures/currents_faults.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()


# Scatter plot of Ia against Ib
plt.figure(figsize=(8, 6))

plt.scatter(
    df[df["fault"] == 0]["Ia"],
    df[df["fault"] == 0]["Ib"],
    label="No Fault",
    alpha=0.6
)

plt.scatter(
    df[df["fault"] == 1]["Ia"],
    df[df["fault"] == 1]["Ib"],
    label="Fault",
    alpha=0.6
)

plt.xlabel("Ia (A)")
plt.ylabel("Ib (A)")
plt.title("Ia vs Ib by Fault Class")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the second figure
plt.savefig(
    "results/figures/current_scatter.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()


# Histogram of Ia for fault and no-fault rows
plt.figure(figsize=(8, 5))

plt.hist(
    df[df["fault"] == 0]["Ia"],
    bins=50,
    alpha=0.6,
    label="No Fault"
)

plt.hist(
    df[df["fault"] == 1]["Ia"],
    bins=50,
    alpha=0.6,
    label="Fault"
)

plt.xlabel("Ia (A)")
plt.ylabel("Number of rows")
plt.title("Distribution of Ia for Fault and No-Fault Conditions")
plt.legend()
plt.grid(True)
plt.tight_layout()

# Save the third figure
plt.savefig(
    "results/figures/current_histogram.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

print("All three EDA figures saved successfully.")
