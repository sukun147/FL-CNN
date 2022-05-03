import socket

class Sever():

    def __init__(self, host:str, port:int, client_num:int):
        self.socket = socket.socket()
        self.socket.bind((host, port))
        self.socket.listen(client_num)
        ...

    def send(self):
        ...

    def recv(self):
        ...

class Client():

    def __init__(self, sever_host:str, sever_port:int):
        self.socket = socket.socket()
        self.socket.connect((sever_host, sever_port))

    def send():
        ...

    def recv():
        ...