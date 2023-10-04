from p2pnetwork.node import Node
import signal
import sys
import time

class InputNode(Node):
    def __init__(self, host, port):
        super().__init__(host, port)

    def node_message(self, node, data):
        if data["type"] == "connect_to_all_nodes":
            # Implement logic to connect to all nodes in the DHT
            pass
        elif data["type"] == "broadcast_data":
            data_value = data["data"]
            print(f"Broadcasting data: {data_value}")
            self.broadcast_message({"type": "send_data", "data": data_value})
            return "Data broadcasted."

def signal_handler(sig, frame):
    print("Input Node shutting down...")
    sys.exit(0)

if __name__ == "__main__":
    host = "127.0.0.1"
    port = 5557
    node = InputNode(host, port)
    node.start()

    # Register the signal handler for Ctrl+C (KeyboardInterrupt)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        while True:
            # Broadcast data every 5 seconds
            node.send_broadcast_message({"type": "broadcast_data", "data": 1.0})
            time.sleep(5)
    except KeyboardInterrupt:
        print("Input Node shutting down...")
        node.stop()
