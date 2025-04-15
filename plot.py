import os
import csv
import numpy as np
import matplotlib.pyplot as plt

envs = ["Pendulum-v1", "BipedalWalker-v3", "LunarLanderContinuous-v2"]
seeds = [555, 666, 777]
alg = "PPO"

for env in envs:
    all_timesteps, all_rewards = [], []

    for seed in seeds:
        path = f'results/{alg}_{env}_seed{seed}.csv'
        if not os.path.exists(path):
            print(f"Missing: {path}")
            continue

        ts, rs = [], []
        with open(path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                ts.append(float(row['timestep']))
                rs.append(float(row['avg_ep_rew']))

        all_timesteps.append(ts)
        all_rewards.append(rs)

    if len(all_rewards) < 1:
        continue

    min_len = min(map(len, all_timesteps))
    avg_ts = np.mean([ts[:min_len] for ts in all_timesteps], axis=0)
    avg_rs = np.mean([rs[:min_len] for rs in all_rewards], axis=0)

    plt.figure()
    plt.plot(avg_ts, avg_rs)
    plt.title(f'{alg} on {env} (avg over {len(all_rewards)} seeds)')
    plt.xlabel("Timesteps")
    plt.ylabel("Avg Episodic Return")
    plt.grid(True)
    plt.savefig(f'results/avg_{alg}_{env}.png', dpi=300)
    plt.close()

    print(f"✅ Saved: results/avg_{alg}_{env}.png")
