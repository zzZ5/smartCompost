#!/bin/bash

# 进入后端目录并启动 Flask
echo "🚀 启动 Flask 后端..."
cd backend
export FLASK_APP=app.py
export FLASK_ENV=development
flask run --host=0.0.0.0 --port=5000 &
FLASK_PID=$!
cd ..

# 进入前端目录并启动 Next.js
echo "🚀 启动 Next.js 前端..."
cd frontend
npm run dev &
FRONT_PID=$!
cd ..

# 等待 Ctrl+C 终止所有后台进程
echo "✅ 前后端启动完成。访问前端: http://localhost:3000"
echo "🔄 后端 API 地址: http://localhost:5000/api/..."
echo "按 Ctrl+C 停止服务"

trap "kill $FLASK_PID $FRONT_PID" EXIT
wait
