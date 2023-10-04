from p2pnetwork.node import Node
import signal
import sys

class OtherNode(Node):
    def __init__(self, host, port):
        super().__init__(host, port)
        self.connected_nodes = []

    def node_message(self, node, data):
        if data["type"] == "get_lowest_distance_node":
            # Replace this logic with actual retrieval of the lowest distance node from the DHT
            lowest_distance_node = {"ip": "127.0.0.1", "port": 5555, "distance": 0.5}
            return lowest_distance_node

        elif data["type"] == "connect_to_node":
            node_info = data["node_info"]
            if len(node_info) == 2:
                self.connected_nodes.append({"ip": node_info[0], "port": node_info[1]})
                return "Connected to node."
            else:
                return "Invalid node info."

        elif data["type"] == "send_data":
            data_value = data["data"]
            print(f"Received data: {data_value}")
            # Perform computation using ReLU
            result = max(float(data_value), 0)
            # Broadcast the result to all connected nodes
            for node in self.connected_nodes:
                self.send_message(node, {"type": "broadcast_result", "result": result})
            return "Data sent and broadcasted."

def signal_handler(sig, frame):
    print("Other Node shutting down...")
    sys.exit(0)
    raise

if __name__ == "__main__":
    host = "127.0.0.1"
    port = 5556
    node = OtherNode(host, port)
    node.start()

    # Register the signal handler for Ctrl+C (KeyboardInterrupt)
    signal.signal(signal.SIGINT, signal_handler)

    try:
        while True:
            pass  # Keep the node running
    except KeyboardInterrupt:
        print("Other Node shutting down...")
        node.stop()
