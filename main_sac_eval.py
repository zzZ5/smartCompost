import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from compost_env_kinetic_physical_continuous import CompostEnvKineticPhysical_Continuous


# ===== 自定义回调函数 =====
class CustomEvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq, save_path, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.save_path = save_path
        self.best_score = -np.inf
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            score = self.evaluate_policy()
            if score > self.best_score:
                self.best_score = score
                model_name = f"custom_best_step{self.num_timesteps}_score{score:.1f}.zip"
                self.model.save(os.path.join(self.save_path, model_name))
                self.model.save(os.path.join(self.save_path, "best_model.zip"))
                print(f"\n✅ 新最优模型保存: {model_name}\n")
        return True

    def evaluate_policy(self):
        obs, _ = self.eval_env.reset()
        done = False
        step = 0
        heating_achieved_step = None
        high_temp_duration = 0
        total_air_on = 0
        O2_min = 100
        O2_good_steps = 0
        O2_too_high_steps = 0
        T_reached_55 = False

        last_vent = False
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            action_value = float(action[0])

            # 三段式判定通风状态
            if action_value >= 0.7:
                vent_on = True
            elif action_value <= 0.3:
                vent_on = False
            else:
                vent_on = last_vent
            last_vent = vent_on

            obs, _, terminated, truncated, _ = self.eval_env.step(action)
            done = terminated or truncated
            T_avg = np.mean(obs[:3])
            O2 = obs[3]
            step = int(obs[7])

            if not T_reached_55 and T_avg >= 55:
                T_reached_55 = True
                if step <= 288:
                    heating_achieved_step = step

            if 55 <= T_avg <= 70:
                high_temp_duration += 1

            O2_min = min(O2_min, O2)
            if 16 <= O2 <= 19:
                O2_good_steps += 1
            if O2 > 19:
                O2_too_high_steps += 1

            if vent_on:
                total_air_on += 1

        if O2_min < 15:
            return -9999

        heating_score = (1 - heating_achieved_step / 288) * 50 if heating_achieved_step else -100
        high_temp_score = (high_temp_duration / (self.eval_env.total_steps - 288)) * 100
        o2_score = (O2_good_steps / self.eval_env.total_steps) * 30 - \
                   (O2_too_high_steps / self.eval_env.total_steps) * 100
        energy_score = max(0, (1 - total_air_on / 840) * 10)
        return heating_score + high_temp_score + o2_score + energy_score


# ===== 主程序入口 =====
def main():
    log_dir = "logs_sac/"
    best_model_dir = "./best_model_sac"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)

    num_cpu = 8
    env = SubprocVecEnv([lambda: CompostEnvKineticPhysical_Continuous() for _ in range(num_cpu)])
    eval_env = CompostEnvKineticPhysical_Continuous()

    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=2e-4,
        batch_size=256,
        gamma=0.99,
        ent_coef="auto",
        device="cpu"
    )

    eval_callback = CustomEvalCallback(
        eval_env=eval_env,
        eval_freq=500,
        save_path=best_model_dir
    )

    model.learn(total_timesteps=1_000_000, callback=eval_callback)

    print("\n📊 训练完成，加载最佳模型并绘图...\n")
    best_model = SAC.load(os.path.join(best_model_dir, "best_model.zip"))
    obs, _ = eval_env.reset()

    T1_list, T2_list, T3_list = [], [], []
    CO2_list, O2_list, H2O_list, actions, steps, rewards = [], [], [], [], [], []

    done = False
    total_reward = 0
    vent_on = False

    while not done:
        action, _ = best_model.predict(obs, deterministic=True)
        a = float(action[0])
        if a >= 0.7:
            vent_on = True
        elif a <= 0.3:
            vent_on = False
        # 否则维持上一次状态

        obs, reward, terminated, truncated, _ = eval_env.step(action)
        done = terminated or truncated

        step = int(obs[7])
        T1_list.append(obs[0])
        T2_list.append(obs[1])
        T3_list.append(obs[2])
        O2_list.append(obs[3])
        CO2_list.append(obs[4])
        H2O_list.append(obs[5])
        actions.append(1 if vent_on else 0)
        steps.append(step)
        rewards.append(reward)
        total_reward += reward

    # 保存 CSV
    with open("run_result_final_sac.csv", "w", newline="") as f:
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
    plt.savefig("run_result_final_sac.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
