"""
Quick test: which ports can this network actually reach on broker.hivemq.com?
Run this once and paste the full output back.
"""
import socket

host = "broker.hivemq.com"
ports_to_test = [1883, 8883, 8000, 8884, 443]

for port in ports_to_test:
    try:
        s = socket.create_connection((host, port), timeout=4)
        print(f"Port {port}: REACHABLE")
        s.close()
    except Exception as e:
        print(f"Port {port}: BLOCKED ({e})")
