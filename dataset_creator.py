import torch
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def generate_data_gpu(n=50000):
    num_sessions = torch.poisson(torch.full((n,), 5.0)).to(device)
    avg_duration = torch.clamp(torch.normal(8, 3, size=(n,)).to(device), 1, 30)
    features_used = torch.clamp((num_sessions + torch.normal(0, 1.5, size=(n,)).to(device)).int(), 1, 10)

    # ✅ Fix: Correct exponential sampling
    exp_dist = torch.distributions.Exponential(rate=1/3.0)
    time_to_first = exp_dist.sample((n,)).to(device)

    notif_clicked = torch.randint(0, 6, (n,), device=device)
    help_accessed = torch.bernoulli(torch.full((n,), 0.3)).int().to(device)
    device_type = torch.randint(0, 2, (n,), device=device)  # 0 = Web, 1 = Mobile
    acquisition_source = torch.multinomial(
        torch.tensor([0.5, 0.3, 0.2], device=device),
        num_samples=n,
        replacement=True
    )

    engagement_score = num_sessions * features_used

    dropoff_score = (
        (num_sessions < 3).int()
        + (avg_duration < 5).int()
        + (features_used < 3).int()
        + (time_to_first > 5).int()
        + (notif_clicked == 0).int()
        + (help_accessed == 0).int()
        + (device_type == 1).int()
    )
    is_dropoff = (dropoff_score >= 4).int()

    df = pd.DataFrame({
        "User_ID": [f"U{i:06d}" for i in range(n)],
        "Num_Sessions_7Days": num_sessions.cpu().numpy(),
        "Avg_Session_Duration": avg_duration.cpu().numpy(),
        "Features_Used_Count": features_used.cpu().numpy(),
        "Time_To_First_Activity": time_to_first.cpu().numpy(),
        "Notifications_Clicked": notif_clicked.cpu().numpy(),
        "Help_Accessed": help_accessed.cpu().numpy(),
        "Device_Type": ["Web" if x == 0 else "Mobile" for x in device_type.cpu().numpy()],
        "Acquisition_Source": [
            ["Organic", "Ad", "Referral"][x] for x in acquisition_source.cpu().numpy()
        ],
        "Engagement_Score": engagement_score.cpu().numpy(),
        "Is_Dropoff": is_dropoff.cpu().numpy()
    })

    return df


# Generate and save
df_gpu = generate_data_gpu(50000)
df_gpu.to_csv("gpu_synthetic_user_data.csv", index=False)
print("✅ Saved dataset to 'gpu_synthetic_user_data.csv'")
