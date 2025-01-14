import socket
import time
import random
import sys
import datetime
import json


class SocketServer():
    def __init__(self, host, port):
        # Create a socket object
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

        # Bind the socket to the host and port
        server_socket.bind((HOST, PORT))

        # Listen for incoming connections
        server_socket.listen(1)
        print('Server is listening on {}:{}'.format(HOST, PORT))

        # Accept a connection from a client
        self.client_socket, client_address = server_socket.accept()
        print('Accepted connection from {}:{}'.format(client_address[0], client_address[1]))


class StreamGenerator():
  def __init__(self, num_streams):
    super().__init__()

  def generate(self, client_socket):
    while True:
        # Generate random data
        data = json.dumps({
            "timestamp": datetime.datetime.now().isoformat(),
            "value": random.randint(0, 100)
        })
        client_socket.send((data + "\n").encode("utf-8"))
        time.sleep(0.1)


if __name__ == "__main__":
    random.seed(57)

    # Define the host and port
    if len(sys.argv) == 3:
        HOST, PORT = sys.argv[1], int(sys.argv[2])
    else:
        HOST, PORT = 'localhost', 9999


    ss = SocketServer(HOST, PORT)
    num_streams = 10
    generator = StreamGenerator(num_streams)
    generator.generate(ss.client_socket)

  