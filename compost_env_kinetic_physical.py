import numpy as np
import gymnasium as gym
from gymnasium import spaces

class CompostEnvKineticPhysical(gym.Env):
    def __init__(self):
        super().__init__()
        self.dt = 600  # 每步10分钟
        self.total_steps = 14 * 24 * 6  # 14天，每小时6步

        # --- 模拟参数 ---
        self.Hc = 50.0            # 有机质单位产热能力
        self.U = 0.12             # 散热系数
        self.heat_capacity = 600  # 热容（影响升温速度）

        # --- 空间定义 ---
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=np.array([0]*3 + [0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([100]*3 + [21, 20, 1, 50, self.total_steps], dtype=np.float32)
        )

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.T = np.array([20.0, 20.0, 20.0])
        self.O2 = 21.0
        self.CO2 = 0.04
        self.H2O = 0.6
        self.room_temp = 25.0
        self.Md = 1000.0
        self.step_num = 0
        self.last_action = None
        self.action_repeat = 0
        self.prev_T = 25.0
        self.prev_O2 = self.O2
        return self._get_obs(), {}

    def step(self, action):
        T_avg = np.mean(self.T)

        # --- 热量计算 ---
        temp_factor = self._temp_factor(T_avg)
        oxygen_factor = self._oxygen_factor()
        heat_gen = self.Hc * (self.Md / 1000.0) * temp_factor * oxygen_factor
        heat_loss = self.U * max(0, T_avg - self.room_temp)
        vent_loss = 0.3 if action == 1 else 0.05
        total_loss = heat_loss + vent_loss
        dT = (heat_gen - total_loss) / (self.heat_capacity * 3)
        self.T += dT * self.dt
        self.T = np.clip(self.T, self.room_temp, 100)

        # --- 氧气变化（强化消耗机制）---
        temp_o2_factor = 1 + 10 * temp_factor
        O2_consume = 0.3 * temp_factor * temp_o2_factor
        self.O2 -= O2_consume
        if action == 1:
            self.O2 = min(21.0, self.O2 + 3.0)
        self.O2 = max(0.0, self.O2)

        # --- CO₂变化 ---
        self.CO2 += 0.05 * O2_consume
        if action == 1:
            self.CO2 *= 0.8
        self.CO2 = np.clip(self.CO2, 0.04, 20)

        # --- 水分蒸发 ---
        humidity_factor = np.clip((self.H2O - 0.2) / 0.4, 0, 1.0)
        evap = min(
            0.0000004 * humidity_factor * max(0, T_avg - self.room_temp) / (self.room_temp + 0.01) *
            (2.0 if action == 1 else 1.0) * self.dt,
            0.0008
        )
        self.H2O -= evap
        self.H2O = np.clip(self.H2O, 0.2, 0.75)

        # --- 有机质消耗 ---
        resp_rate = 200 * O2_consume * temp_factor
        self.Md -= resp_rate * self.dt / 3600
        self.Md = max(0.0, self.Md)

        # --- 动作记录 ---
        if self.last_action == action:
            self.action_repeat += 1
        else:
            self.action_repeat = 0
        self.last_action = action

        self.step_num += 1
        done = self.step_num >= self.total_steps
        truncated = False



        # === 强制终止条件：O₂ 过低 ===
        if self.O2 < 15:
            print(f"[终止] O₂过低：{self.O2:.2f}%，Step={self.step_num}")
            return self._get_obs(), -100000, True, False, {}

        # --- 分阶段奖励函数 ---
        phase = self._get_phase(self.step_num, T_avg)
        reward = 0

        if phase == "heating":
            delta_T = T_avg - self.prev_T
            reward += max(0, delta_T) * 5.0  # 奖励升温

        elif phase == "high":
            if T_avg >= 55:
                reward += 3.0

        elif phase == "cooling":
            if self.O2 > 19.5 and action == 1:
                reward -= 0.5


        # 氧气控制奖励
        if self.O2 < 18 and action == 0:  # 低于18时，惩罚
            reward -= (18 - self.O2)**2 *10
        if 18 <= self.O2 <= 19.5:
            reward += 1.0  # 鼓励维持在理想区
        if self.O2 >= 19.5 and action == 1:  # 大于19.5时曝气惩罚 
            reward -= 1.0

        # --- 其他奖励 ---
        reward += (0.6 - self.H2O) * 1  # 鼓励水分下降        
        reward += max(0, 1.0 - self.CO2) * 0.5  # CO₂控制奖励

        if self.action_repeat == 0:
            reward += 1  # 鼓励操作多样性
        elif self.action_repeat > 20:
            reward -= 1 + 0.05 * (self.action_repeat - 20)


        self.prev_T = T_avg
        self.prev_O2 = self.O2

        if self.step_num % 30 == 0:
            print(f"[{self.step_num}] 阶段={phase}, T={T_avg:.1f}°C, O₂={self.O2:.2f}%, H₂O={self.H2O:.2f}, A={action}, Md={self.Md:.2f}, R={reward:.2f}")

        return self._get_obs(), reward, done, truncated, {}

    def _get_obs(self):
        return np.array([
            *self.T, self.O2, self.CO2, self.H2O, self.room_temp, self.step_num
        ], dtype=np.float32)

    def _temp_factor(self, T):
        s1 = 1 / (1 + np.exp(-(T - 43)/7))
        s2 = 1 / (1 + np.exp((T - 70)/5))
        return s1 * s2 * 0.15

    def _oxygen_factor(self):
        return np.clip(self.O2 / 21.0, 0.0, 1.0)

    def _get_phase(self, step_num, T_avg):
        if step_num < 288 and T_avg < 55:
            return "heating"
        elif 288 <= step_num < 1008 :
            return "high"
        else:
            if T_avg < 45:
                return "cooling"
            else:
                return "high"
