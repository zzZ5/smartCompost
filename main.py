import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from compost_env_kinetic_physical import CompostEnvKineticPhysical


# ========== 自定义回调函数：基于综合控制指标判断最优模型 ==========
class CustomEvalCallback(BaseCallback):
    """
    自定义模型评估回调，用于在训练过程中根据升温速度、高温持续时长、氧气控制和节能情况综合评估模型，
    保存表现最好的模型。
    """
    def __init__(self, eval_env, eval_freq, save_path, verbose=1):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.save_path = save_path
        self.best_score = -np.inf
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self):
        # 每 eval_freq 步评估一次
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
        """
        策略评估函数：模拟一次完整堆肥过程，计算综合评分。
        """
        obs, _ = self.eval_env.reset()
        done = False
        step = 0

        # 各项指标初始化
        heating_achieved_step = None       # 升温至55°C的首次时间
        high_temp_duration = 0             # 高温阶段持续时间
        total_air_on = 0                   # 通风总次数
        O2_min = 100                       # 最低氧气
        O2_good_steps = 0                 # O2 在16~19% 区间步数
        O2_too_high_steps = 0             # O2 > 19% 步数
        T_reached_55 = False

        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = self.eval_env.step(action)
            done = terminated or truncated
            T_avg = np.mean(obs[:3])
            O2 = obs[3]
            step = int(obs[7])

            # 判断是否在前48小时升至 55°C
            if not T_reached_55 and T_avg >= 55:
                T_reached_55 = True
                if step <= 288:
                    heating_achieved_step = step

            # 一旦温度达到 55°C，即进入高温判定阶段
            if T_avg >= 55 and T_avg <= 70:
                high_temp_duration += 1

            # 气体控制
            O2_min = min(O2_min, O2)
            if 16 <= O2 <= 19:
                O2_good_steps += 1
            if O2 > 19:
                O2_too_high_steps += 1

            # 通风次数
            if action == 1:
                total_air_on += 1

        # -------- 综合评分计算 --------
        # 1. 升温期得分：越早升温越高分（满分 50）
        if heating_achieved_step:
            heating_score = (1 - heating_achieved_step / 288) * 50
        else:
            heating_score = -100  # 没有升温

        # 2. 高温持续得分：满分 100
        high_temp_score = (high_temp_duration / (self.eval_env.total_steps - 288)) * 100

        # 3. 氧气得分（O2 不低于15%，16~19区间步数占比高）
        if O2_min < 15:
            o2_score = -1000  # 严重缺氧
        else:
            o2_score = (O2_good_steps / self.eval_env.total_steps) * 30
            o2_score -= (O2_too_high_steps / self.eval_env.total_steps) * 100  # O2 > 19 惩罚

        # 4. 节能得分：通风越少越高，最多840步，满分10
        energy_score = max(0, (1 - total_air_on / 840) * 10)

        # 5. 总分
        total_score = heating_score + high_temp_score + o2_score + energy_score
        return total_score


# ========== 主程序入口 ==========
def main():
    # 创建日志与模型保存目录
    log_dir = "logs/"
    best_model_dir = "./best_model"
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(best_model_dir, exist_ok=True)

    # 初始化训练环境与评估环境
    num_cpu = 9
    env = SubprocVecEnv([lambda: CompostEnvKineticPhysical() for _ in range(num_cpu)])
    eval_env = CompostEnvKineticPhysical()

    # 初始化 PPO 模型
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=log_dir,
        learning_rate=2e-4,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        ent_coef=0.3,
        device="cpu"
    )

    # 定义回调函数：按指标保存最优模型
    eval_callback = CustomEvalCallback(
        eval_env=eval_env,
        eval_freq=5000,
        save_path=best_model_dir
    )

    # 开始训练
    model.learn(total_timesteps=10_000_000, callback=eval_callback)

    # ===== 训练结束后评估最优模型 =====
    print("\n📊 训练完成，加载最佳模型并绘图...\n")
    best_model = PPO.load(os.path.join(best_model_dir, "best_model.zip"))
    obs, _ = eval_env.reset()

    # 初始化结果列表
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

    # 保存CSV结果文件
    with open("run_result_final.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Hour", "T1", "T2", "T3", "O2", "CO2", "H2O", "Action", "Reward"])
        for i in range(len(T1_list)):
            hour = float(steps[i]) * 10 / 60
            writer.writerow([hour, T1_list[i], T2_list[i], T3_list[i],
                             O2_list[i], CO2_list[i], H2O_list[i], actions[i], rewards[i]])

    # 结果可视化绘图
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
