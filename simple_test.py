#!/usr/bin/env python3
import asyncio
import websockets

async def simple_test():
    uri = "ws://localhost:8000/ws"
    try:
        print("Attempting to connect...")
        async with websockets.connect(uri) as websocket:
            print("Connected successfully!")
            await asyncio.sleep(1)
            print("Connection test completed")
    except Exception as e:
        print(f"Connection failed: {e}")

if __name__ == "__main__":
    asyncio.run(simple_test()) 