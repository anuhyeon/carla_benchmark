def send(socket, message):
    socket.send_pyobj(message)


def receive(socket):
    return socket.recv_pyobj()
