from flask import Flask, jsonify, request
from mqtt_hander import start_mqtt
from database import init_db, get_latest_data, get_recent_data, get_stats

app = Flask(__name__)

# 初始化数据库
init_db()

# 启动 MQTT 监听
start_mqtt()


@app.route("/")
def index():
    return "✅ Smart Compost Flask Server is Running"


# ✅ 获取最新一条数据
@app.route("/api/latest", methods=["GET"])
def api_latest():
    data = get_latest_data()
    return jsonify(data)


# ✅ 获取最近 N 条数据
@app.route("/api/history", methods=["GET"])
def api_history():
    try:
        limit = int(request.args.get("limit", 50))
    except ValueError:
        limit = 50
    data = get_recent_data(limit)
    return jsonify(data)


# ✅ 获取统计信息（可扩展）
@app.route("/api/stats", methods=["GET"])
def api_stats():
    stats = get_stats()
    return jsonify(stats)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
