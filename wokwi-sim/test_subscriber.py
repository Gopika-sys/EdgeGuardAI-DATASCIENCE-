"""
EdgeGuard AI - MQTT Subscriber Test Script

Run this on your laptop BEFORE building the full FastAPI listener, just to
confirm the Wokwi simulation is actually publishing real data to the broker.

Usage:
    pip install paho-mqtt
    python test_subscriber.py

It subscribes to every sensor topic under truck1 and prints each message as
it arrives. Ctrl+C to stop.
"""

import json
from datetime import datetime
import paho.mqtt.client as mqtt

BROKER = "broker.hivemq.com"
PORT = 1883
TOPIC_FILTER = "edgeguard/truck1/#"  # all sensors + status for truck1


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"[connected] {BROKER}:{PORT}  subscribing to {TOPIC_FILTER}")
        client.subscribe(TOPIC_FILTER)
    else:
        print(f"[error] connection failed, rc={rc}")


def on_message(client, userdata, msg):
    ts = datetime.now().strftime("%H:%M:%S")
    try:
        payload = json.loads(msg.payload.decode())
        sensor = payload.get("sensor", "?")
        value = payload.get("value", "?")
        unit = payload.get("unit", "")
        print(f"[{ts}] {msg.topic:45s} {sensor:22s} {value} {unit}")
    except json.JSONDecodeError:
        print(f"[{ts}] {msg.topic:45s} RAW: {msg.payload.decode()}")


def main():
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message

    print("Connecting...")
    client.connect(BROKER, PORT, keepalive=60)
    client.loop_forever()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped.")
