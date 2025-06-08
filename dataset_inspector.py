import torch
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
#import matplotlib.font_manager as fm
plt.rcParams['font.family'] = 'Segoe UI Emoji'

# Load CSV
FILENAME = "gpu_synthetic_user_data.csv"
assert os.path.exists(FILENAME), f"File '{FILENAME}' not found."
# Create output directory if it doesn't exist
os.makedirs("Observations", exist_ok=True)
df = pd.read_csv(FILENAME)
print("✅ Dataset loaded with shape:", df.shape)

# Send numeric data to GPU for fast stat analysis
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Numeric columns for GPU-based stats
numeric_cols = [
    "Num_Sessions_7Days", "Avg_Session_Duration", "Features_Used_Count",
    "Time_To_First_Activity", "Notifications_Clicked", "Help_Accessed",
    "Engagement_Score"
]

# Convert to tensors
tensor_data = torch.tensor(df[numeric_cols].values, dtype=torch.float32).to(device)

# Summary stats using GPU
mean_vals = torch.mean(tensor_data, dim=0)
std_vals = torch.std(tensor_data, dim=0)
min_vals = torch.min(tensor_data, dim=0).values
max_vals = torch.max(tensor_data, dim=0).values

print("\n📊 Numeric Column Stats:")
for i, col in enumerate(numeric_cols):
    print(f"{col:<25}  mean={mean_vals[i]:.2f}  std={std_vals[i]:.2f}  min={min_vals[i]:.2f}  max={max_vals[i]:.2f}")

# Class balance
drop_counts = df["Is_Dropoff"].value_counts()
print(f"\n⚖️ Class Balance (Dropoff):\n{drop_counts.to_string()}")
print(f"Dropoff Rate: {drop_counts.get(1, 0) / df.shape[0]:.2%}")

# Correlation matrix
corr = df[numeric_cols + ["Is_Dropoff"]].corr()

plt.figure(figsize=(10, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("🔗 Correlation Heatmap")
plt.tight_layout()
plt.savefig("Observations/correlation_heatmap.png")
print("✅ Saved heatmap as 'correlation_heatmap.png'")

# Device vs Dropoff
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Device_Type", hue="Is_Dropoff")
plt.title("📱 Dropoff by Device Type")
plt.tight_layout()
plt.savefig("Observations/dropoff_by_device.png")

# Acquisition Source vs Dropoff
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x="Acquisition_Source", hue="Is_Dropoff")
plt.title("🌐 Dropoff by Acquisition Source")
plt.tight_layout()
plt.savefig("Observations/dropoff_by_source.png")

# Boxplots for selected features
important_features = ["Num_Sessions_7Days", "Avg_Session_Duration", "Features_Used_Count", "Time_To_First_Activity"]

for feature in important_features:
    plt.figure(figsize=(6, 4))
    sns.boxplot(data=df, x="Is_Dropoff", y=feature)
    plt.title(f"📦 {feature} vs Dropoff")
    plt.tight_layout()
    plt.savefig(f"Observations/{feature}_vs_dropoff.png")

print("📁 Saved all visualizations.")

# Feature Target Correlation (sorted)
target_corr = corr["Is_Dropoff"].drop("Is_Dropoff").sort_values(ascending=False)
print("\n🔎 Feature correlations with Dropoff:\n", target_corr)

# Optional: Check unrealistic patterns
if (df["Num_Sessions_7Days"] == 0).sum() > 0:
    print("⚠️ Users with 0 sessions exist!")
if (df["Avg_Session_Duration"] > 60).any():
    print("⚠️ Some session durations > 60 mins — check realism")

print("\n✅ Dataset profiling complete.")
