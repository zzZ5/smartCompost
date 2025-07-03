import os
import json
import time
import paho.mqtt.client as mqtt
from datetime import datetime
import numpy as np
from stable_baselines3 import SAC
from database import save_data  # 需包含 fan、p 值字段

# ==== 模型加载 ====
MODEL_PATH = "models/best_model.zip"
model = SAC.load(MODEL_PATH)

# ==== MQTT 配置 ====
MQTT_BROKER = "118.25.108.254"
MQTT_PORT = 1883
POST_TOPIC = "compostlab/KgSERnY2Zn/post/data"
RESPONSE_TOPIC = "compostlab/KgSERnY2Zn/response"

# ==== Key 与变量映射 ====
SENSOR_KEYS = {
    "CO2": "ue5RkzOpT5jGaQR",
    "O2": "Z2R5Ep3GP4pB1tk",
    "Temp": "RIbjMs78qOdtr0z",
    "Mois": "iu60u9UCTMk2gY7",
}


def predict_fan_action(model, T1, T2, T3, O2, CO2, H2O):
    obs = np.array([T1, T2, T3, O2, CO2, H2O, 0.0, 0.0], dtype=np.float32).reshape(
        1, -1
    )
    p, _ = model.predict(obs, deterministic=True)
    p_value = float(p[0])

    if p_value > 0.7:
        return "on", p_value
    elif p_value < 0.3:
        return "off", p_value
    else:
        return "hold", p_value


def send_fan_command(action):
    if action in ("on", "off"):
        cmd_payload = {
            "device": "KgSERnY2Zn",
            "commands": [{"command": "aeration", "action": action}],
        }
        mqtt_client.publish(RESPONSE_TOPIC, json.dumps(cmd_payload), qos=0)
        print(f"📤 已发送风机控制指令: {action}")


def parse_sensor_data(data_list):
    parsed = {}
    for item in data_list:
        key = item.get("key")
        val = item.get("value")
        for name, k in SENSOR_KEYS.items():
            if key == k:
                parsed[name] = val
    return parsed


# ==== MQTT 回调 ====
def on_connect(client, userdata, flags, rc):
    print("✅ 已连接至 MQTT Broker")
    client.subscribe(POST_TOPIC)
    print(f"📡 已订阅主题：{POST_TOPIC}")


def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        data_list = payload.get("data", [])
        parsed = parse_sensor_data(data_list)

        if all(k in parsed for k in ("CO2", "O2", "Temp", "Mois")):
            CO2 = parsed["CO2"]
            O2 = parsed["O2"]
            Temp = parsed["Temp"]
            H2O = parsed["Mois"]

            # 模拟反应器内部温度（如无 T1~T3 传感器）
            T1 = T2 = T3 = Temp

            action, p_value = predict_fan_action(model, T1, T2, T3, O2, CO2, H2O)
            send_fan_command(action)

            save_data(
                {
                    "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "T1": T1,
                    "T2": T2,
                    "T3": T3,
                    "O2": O2,
                    "CO2": CO2,
                    "H2O": H2O,
                    "fan": action,
                    "p_value": round(p_value, 3),
                }
            )

            print(f"📥 数据处理成功：action={action}, p={p_value:.3f}")

        else:
            print("⚠️ 传感器数据不完整，跳过")

    except Exception as e:
        print(f"❌ MQTT 消息处理错误: {e}")


# ==== 启动客户端 ====
def start_mqtt():
    global mqtt_client
    mqtt_client = mqtt.Client()
    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message
    mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
    mqtt_client.loop_start()
