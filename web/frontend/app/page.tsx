'use client'

import React, { useEffect, useState } from 'react'
import axios from 'axios'
import {
  LineChart, Line, CartesianGrid, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer
} from 'recharts'

type DataPoint = {
  time: string
  T1: number
  T2: number
  T3: number
  O2: number
  CO2: number
  H2O: number
  fan: number
}

export default function Dashboard() {
  const [data, setData] = useState<DataPoint[]>([])
  const [limit, setLimit] = useState<number>(100)

  useEffect(() => {
    const fetchData = async () => {
      try {
        const res = await axios.get(`http://localhost:5000/api/history?limit=${limit}`)
        setData(res.data)
      } catch (err) {
        console.error('❌ 数据获取失败:', err)
      }
    }

    fetchData()
    const timer = setInterval(fetchData, 10000)
    return () => clearInterval(timer)
  }, [limit]) // limit 改变时重新 fetch

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-4">🌱 Smart Compost Dashboard</h1>

      {/* Limit 控制面板 */}
      <div className="mb-6 flex items-center space-x-4">
        <label htmlFor="limit" className="font-medium">显示条数:</label>
        <input
          id="limit"
          type="number"
          min={10}
          max={1000}
          value={limit}
          onChange={(e) => setLimit(Number(e.target.value))}
          className="border px-3 py-1 rounded w-24"
        />
        <span className="text-sm text-gray-500">（每10秒自动刷新）</span>
      </div>

      <div className="grid grid-cols-1 gap-6">
        <Chart title="Average Temperature (°C)" data={data} dataKey="T1" color="#0077cc" />
        <Chart title="Oxygen (%)" data={data} dataKey="O2" color="#228B22" />
        <Chart title="CO₂ (%)" data={data} dataKey="CO2" color="#ff7300" />
        <Chart title="Moisture (H₂O %)" data={data} dataKey="H2O" color="#00bfff" />
        <Chart title="Fan On/Off" data={data} dataKey="fan" color="#333" isStep />
      </div>
    </div>
  )
}

function Chart({ title, data, dataKey, color, isStep = false }: {
  title: string
  data: DataPoint[]
  dataKey: string
  color: string
  isStep?: boolean
}) {
  return (
    <div>
      <h2 className="text-lg font-semibold mb-2">{title}</h2>
      <ResponsiveContainer width="100%" height={250}>
        <LineChart data={data}>
          {isStep ? (
            <Line type="stepAfter" dataKey={dataKey} stroke={color} dot={false} />
          ) : (
            <Line type="monotone" dataKey={dataKey} stroke={color} dot={false} />
          )}
          <CartesianGrid stroke="#ccc" />
          <XAxis dataKey="time" />
          <YAxis />
          <Tooltip />
          <Legend />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
