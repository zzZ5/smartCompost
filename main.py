import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import shutil

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv
from compost_env_kinetic_physical import CompostEnvKineticPhysical

class SaveAllBestModelsCallback(EvalCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._last_best_reward = -np.inf

    def _on_step(self):
        result = super()._on_step()
        if result and self.best_mean_reward > self._last_best_reward:
            self._last_best_reward = self.best_mean_reward
            step_count = self.num_timesteps
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            model_name = f"model_step{step_count}_reward{self._last_best_reward:.2f}_{timestamp}.zip"
            model_path = os.path.join(self.best_model_save_path, model_name)
            shutil.copyfile(os.path.join(self.best_model_save_path, "best_model.zip"), model_path)
            print(f"\n✅ 保存最优模型：{model_name}\n")
        return result

def main():
    log_dir = "logs/"
    os.makedirs(log_dir, exist_ok=True)
    best_model_dir = "./best_model"
    os.makedirs(best_model_dir, exist_ok=True)

    num_cpu = 9
    env = SubprocVecEnv([lambda: CompostEnvKineticPhysical() for _ in range(num_cpu)])
    eval_env = CompostEnvKineticPhysical()

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=1e-4,
        n_steps=4096,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.5,
        device="cpu"
    )

    eval_callback = SaveAllBestModelsCallback(
        eval_env,
        best_model_save_path=best_model_dir,
        log_path=log_dir,
        eval_freq=20000,
        n_eval_episodes=20,
        deterministic=True,
        render=False
    )

    model.learn(total_timesteps=1_000_000, callback=eval_callback)

    print("\n📊 训练完成，开始绘图评估最终模型...\n")

    best_model = PPO.load(os.path.join(best_model_dir, "best_model.zip"))
    obs, _ = eval_env.reset()

    T1_list, T2_list, T3_list = [], [], []
    CO2_list, O2_list, H2O_list, actions, steps, rewards = [], [], [], [], [], []
    done = False
    total_reward = 0

    while not done:
        action, _ = best_model.predict(obs, deterministic=True)
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
        total_reward += reward

        if step % 50 == 0:
            T_avg = np.mean(obs[:3])
            print(f"[{step * 10 / 60:.1f} h] Tavg={T_avg:.1f} °C | O₂={obs[3]:.2f}% | CO₂={obs[4]:.2f}% | H₂O={obs[5]:.2f} | A={action} | R={reward:.2f}")

    # 保存 CSV
    csv_name = "run_result_final.csv"
    with open(csv_name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Hour", "T1", "T2", "T3", "O2", "CO2", "H2O", "Action", "Reward"])
        for i in range(len(T1_list)):
            hour = float(steps[i]) * 10 / 60
            writer.writerow([hour, T1_list[i], T2_list[i], T3_list[i],
                             O2_list[i], CO2_list[i], H2O_list[i], actions[i], rewards[i]])

    # 绘图
    T_avg = np.mean([T1_list, T2_list, T3_list], axis=0)
    time = [s * 10 / 60 for s in steps]

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
    plt.plot(time, rewards, label="Reward", color="purple")
    plt.ylabel(f"Reward\nTotal={total_reward:.1f}")
    plt.xlabel("Time (hour)")
    plt.legend()

    plt.tight_layout()
    plt.savefig("run_result_final.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
