import os
import re
import csv
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from compost_env_kinetic_physical import CompostEnvKineticPhysical

def evaluate_and_plot(model_path, save_dir="logs"):
    model = PPO.load(model_path)
    eval_env = CompostEnvKineticPhysical()
    obs, _ = eval_env.reset()

    T1_list, T2_list, T3_list = [], [], []
    CO2_list, O2_list, H2O_list, actions, steps, rewards = [], [], [], [], [], []
    done = False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = eval_env.step(action)
        done = terminated or truncated

        step = int(obs[7])
        T1_list.append(obs[0])
        T2_list.append(obs[1])
        T3_list.append(obs[2])
        O2_list.append(obs[3])
        CO2_list.append(obs[4])
        H2O_list.append(obs[5])
        actions.append(int(action))
        steps.append(step)
        rewards.append(reward)

    model_name = os.path.splitext(os.path.basename(model_path))[0]
    csv_path = os.path.join(save_dir, f"run_result_{model_name}.csv")
    png_path = os.path.join(save_dir, f"run_result_{model_name}.png")

    os.makedirs(save_dir, exist_ok=True)

    # 保存 CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Hour", "T1", "T2", "T3", "O2", "CO2", "H2O", "Action", "Reward"])
        for i in range(len(T1_list)):
            hour = float(steps[i]) * 10 / 60
            writer.writerow([hour, T1_list[i], T2_list[i], T3_list[i],
                            O2_list[i], CO2_list[i], H2O_list[i], actions[i], rewards[i]])

    # 绘图
    T_avg = np.mean([T1_list, T2_list, T3_list], axis=0)
    time = [s * 10 / 60 for s in steps]
    total_reward = sum(rewards)

    plt.figure(figsize=(12, 12))

    plt.subplot(6, 1, 1)
    plt.plot(time, T_avg, label="Avg Temperature (°C)", color="steelblue")
    plt.axhline(55, linestyle="--", color="gray", label="Target 55°C")
    plt.ylabel("T avg (°C)")
    plt.legend()

    plt.subplot(6, 1, 2)
    plt.plot(time, O2_list, label="Oxygen (%)", color="green")
    plt.axhline(18, linestyle="--", color="red", label="O₂ Warning")
    plt.axhline(15, linestyle="--", color="darkred", label="O₂ Critical")
    plt.ylabel("O₂ (%)")
    plt.legend()

    plt.subplot(6, 1, 3)
    plt.plot(time, CO2_list, label="CO₂ (%)", color="orange")
    plt.axhline(1.0, linestyle="--", color="gray")
    plt.ylabel("CO₂ (%)")
    plt.legend()

    plt.subplot(6, 1, 4)
    plt.plot(time, H2O_list, label="Moisture (%)", color="blue")
    plt.axhline(0.6, linestyle="--", color="gray")
    plt.ylabel("H₂O (%)")
    plt.legend()

    plt.subplot(6, 1, 5)
    plt.step(time, actions, label="Action (0=off, 1=on)", color="black", where='post')
    plt.ylabel("Action")
    plt.legend()

    plt.subplot(6, 1, 6)
    plt.plot(time, rewards, label=f"Reward (sum={total_reward:.1f})", color="purple")
    plt.ylabel("Reward")
    plt.xlabel("Time (hour)")
    plt.legend()

    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.close()
    print(f"✅ 绘图完成：{png_path}")


def main():
    model_dir = "./best_model"
    save_dir = "./logs"

    model_files = [
        f for f in os.listdir(model_dir)
        if f.startswith("custom_best_step") and f.endswith(".zip")
    ]

    # 按 step 排序
    def extract_step(name):
        match = re.search(r"step(\d+)", name)
        return int(match.group(1)) if match else float('inf')

    model_files = sorted(model_files, key=extract_step)

    if not model_files:
        print("⚠️ 未找到任何模型文件。请检查 ./best_model 是否包含 model_step*.zip 文件。")
        return

    for model_file in model_files:
        full_path = os.path.join(model_dir, model_file)
        evaluate_and_plot(full_path, save_dir=save_dir)


if __name__ == "__main__":
    main()
