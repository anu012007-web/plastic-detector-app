# webhook_receiver.py
"""
EcoScanIndia - Webhook Receiver Server
Exposes HTTP endpoints for the Raspberry Pi robot to post rain shield status updates.
Saves updates to the central SQLite database (detections.db) for the Streamlit dashboard.
"""

import sqlite3
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from typing import Optional

app = FastAPI(title="EcoScanIndia Robot Webhook Receiver")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class RobotStatusPayload(BaseModel):
    battery: int              # 0 to 100
    shield_position: str     # "RETRACTED" or "DEPLOYED"
    rain_sensor: str         # "DRY" or "WET"
    is_raining: bool
    is_docked: bool
    mode: str                # "AUTO" or "MANUAL_OVERRIDE"
    temperature: Optional[float] = None
    humidity: Optional[float] = None

def init_db():
    """Ensures the robot_status table is initialized in detections.db with the updated schema."""
    conn = sqlite3.connect('detections.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS robot_status
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT,
                  battery INTEGER,
                  shield_position TEXT,
                  rain_sensor TEXT,
                  is_raining INTEGER,
                  is_docked INTEGER,
                  mode TEXT,
                  temperature REAL,
                  humidity REAL)''')
    conn.commit()
    conn.close()

# Initialize DB on startup
init_db()

@app.post("/api/robot/status")
async def update_robot_status(status: RobotStatusPayload):
    """Receives POST updates from the robot and logs them to detections.db."""
    try:
        conn = sqlite3.connect('detections.db')
        c = conn.cursor()
        c.execute("""
            INSERT INTO robot_status 
            (timestamp, battery, shield_position, rain_sensor, is_raining, is_docked, mode, temperature, humidity) 
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            status.battery,
            status.shield_position,
            status.rain_sensor,
            1 if status.is_raining else 0,
            1 if status.is_docked else 0,
            status.mode,
            status.temperature,
            status.humidity
        ))
        conn.commit()
        conn.close()
        print(f"[API] Status Logged: Mode={status.mode}, Shield={status.shield_position}, Rain={status.rain_sensor}, Battery={status.battery}%")
        return {"status": "success", "message": "Robot status updated successfully."}
    except Exception as e:
        print(f"[API ERROR] Failed to log robot status: {e}")
        raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")

@app.get("/api/robot/status")
async def get_latest_status():
    """Exposes GET endpoint to fetch the latest robot status."""
    try:
        conn = sqlite3.connect('detections.db')
        c = conn.cursor()
        c.execute("""
            SELECT timestamp, battery, shield_position, rain_sensor, is_raining, is_docked, mode, temperature, humidity 
            FROM robot_status 
            ORDER BY id DESC LIMIT 1
        """)
        row = c.fetchone()
        conn.close()
        
        if row:
            return {
                "timestamp": row[0],
                "battery": row[1],
                "shield_position": row[2],
                "rain_sensor": row[3],
                "is_raining": bool(row[4]),
                "is_docked": bool(row[5]),
                "mode": row[6],
                "temperature": row[7],
                "humidity": row[8]
            }
        else:
            return {"status": "idle", "message": "No robot status reports received yet."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print("Starting EcoScanIndia Webhook Receiver on port 8000...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
