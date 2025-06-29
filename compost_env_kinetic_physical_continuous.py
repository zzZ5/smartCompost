import numpy as np
import gymnasium as gym
from gymnasium import spaces

class CompostEnvKineticPhysical_Continuous(gym.Env):
    def __init__(self):
        super().__init__()
        self.dt = 600  # 每步10分钟
        self.total_steps = 14 * 24 * 6  # 14天，每小时6步

        # 模拟参数
        self.Hc = 50.0
        self.U = 0.12
        self.heat_capacity = 600

        # 连续动作空间（0~1之间）
        self.action_space = spaces.Box(low=np.array([0.0]), high=np.array([1.0]), dtype=np.float32)

        # 观测空间：3个温度 + O2 + CO2 + 水分 + 室温 + 步数（无阶段码）
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
        self.high_temp_steps = 0
        self.vent_state = False
        return self._get_obs(), {}

    def step(self, action):
        action = np.clip(action[0], 0.0, 1.0)

        if action >= 0.7:
            self.vent_state = True
        elif action <= 0.3:
            self.vent_state = False

        T_avg = np.mean(self.T)
        temp_factor = self._temp_factor(T_avg)
        oxygen_factor = self._oxygen_factor()
        heat_gen = self.Hc * (self.Md / 1000.0) * temp_factor * oxygen_factor
        heat_loss = self.U * max(0, T_avg - self.room_temp)
        vent_loss = 0.4 if self.vent_state else 0.05
        total_loss = heat_loss + vent_loss
        dT = (heat_gen - total_loss) / (self.heat_capacity * 3)
        self.T += dT * self.dt
        self.T = np.clip(self.T, self.room_temp, 100)

        temp_o2_factor = 1 + 10 * temp_factor
        O2_consume = 0.3 * temp_factor * temp_o2_factor
        self.O2 -= O2_consume * 5
        if self.vent_state:
            self.O2 = min(21.0, self.O2 + 2.0)
        self.O2 = max(0.0, self.O2)

        self.CO2 += 0.05 * O2_consume
        if self.vent_state:
            self.CO2 *= 0.8
        self.CO2 = np.clip(self.CO2, 0.04, 20)

        humidity_factor = np.clip((self.H2O - 0.2) / 0.4, 0, 1.0)
        evap = min(
            0.0000004 * humidity_factor * max(0, T_avg - self.room_temp) / (self.room_temp + 0.01) *
            (2.0 if self.vent_state else 1.0) * self.dt,
            0.0008
        )
        self.H2O -= evap
        self.H2O = np.clip(self.H2O, 0.2, 0.75)

        resp_rate = 200 * O2_consume * temp_factor
        self.Md -= resp_rate * self.dt / 3600
        self.Md = max(0.0, self.Md)

        if self.last_action == self.vent_state:
            self.action_repeat += 1
        else:
            self.action_repeat = 0
        self.last_action = self.vent_state

        self.step_num += 1
        done = self.step_num >= self.total_steps
        truncated = False

        phase = self._get_phase(self.step_num, T_avg)
        reward = 0

        if phase == "heating":
            delta_T = T_avg - self.prev_T
            reward += delta_T * 20.0 if delta_T > 0 else -10.0
        elif phase == "high":
            if 55 <= T_avg <= 70:
                reward += 2.0
                self.high_temp_steps += 1
                reward += min(self.high_temp_steps * 0.2, 20.0)
            else:
                self.high_temp_steps = 0
        elif phase == "cooling":
            delta_T = self.prev_T - T_avg
            reward += 2.0 if 0 <= delta_T <= 1 else -max((delta_T - 1) * 5.0, 0)

        reward += self._oxygen_reward(self.O2, int(self.vent_state), phase)
        reward += (0.6 - self.H2O) * 1.5
        reward += max(0, 1.0 - self.CO2) * 1.0

        if self.action_repeat == 0:
            reward += 2.0
        elif self.action_repeat > 6:
            reward -= 0.5 * (self.action_repeat - 6)

        self.prev_T = T_avg
        self.prev_O2 = self.O2

        if self.step_num % 30 == 0:
            print(f"[{self.step_num}] 阶段={phase}, T={T_avg:.1f}°C, O₂={self.O2:.2f}%, H₂O={self.H2O:.2f}, 风机={int(self.vent_state)}, Md={self.Md:.2f}, R={reward:.2f}")

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
        elif T_avg >= 55:
            return "high"
        elif T_avg < 45:
            return "cooling"
        else:
            return "high"

    def _oxygen_reward(self, o2, action, phase):
        if phase in ["heating", "cooling"]:
            if o2 < 16:
                return -(17 - o2)**2 * 10 if action == 0 else -10
            elif 16 <= o2 <= 19: 
                return (20 - o2) * 5 if action == 1 else 5
            elif o2 > 19 and action == 1: 
                return -(o2 - 19) * 5
        elif phase == "high":
            if o2 < 18: 
                return -(18 - o2)**2 * 10
            if 18 <= o2 <= 20: 
                return 2
            if o2 > 20: 
                return -(o2 - 20) * 5
        return 0
