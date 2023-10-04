from p2pnetwork.node import Node
import signal
import sys

class DHTNode(Node):
    def __init__(self, host, port, distance):
        super().__init__(host, port)
        self.distance = distance

    def node_message(self, node, data):
        if data["type"] == "get_info":
            return {"ip": self.host, "port": self.port, "distance": self.distance}

    def node_connection(self, node):
        print(f"Connected to node: {node.host}:{node.port}")

    def node_disconnection(self, node):
        print(f"Disconnected from node: {node.host}:{node.port}")

def signal_handler(sig, frame):
    print("DHT Node shutting down...")
    sys.exit(0)
    raise

if __name__ == "__main__":
    host = "127.0.0.1"
    port = 5555
    distance = 0.3
    node = DHTNode(host, port, distance)
    node.start()

    # Register the signal handler for Ctrl+C (KeyboardInterrupt)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        while True:
            pass  # Keep the node running
    except KeyboardInterrupt:
        print("DHT Node shutting down...")
        node.stop()
