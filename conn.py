import socket
import pickle


class Server:

    def __init__(self, host: str, port: int, client_num: int):
        self.socket = socket.socket()
        self.socket.bind((host, port))
        self.socket.listen(client_num)
        client, port = self.socket.accept()
        self.client = client

    def send(self, obj: object) -> bool:
        """
        :param obj: 需要发送的对象
        :return: 成功返回True，失败返回False
        """
        try:
            data = pickle.dumps(obj)
            size = len(data)
            self.client.send(str(size).encode())  # 报头
            self.client.send(data)
            return True
        except Exception as e:
            print('Error:', e)
            return False

    def recv(self):
        """
        :return: 成功返回接收到的对象并反序列化，失败返回None
        """
        try:
            is_conn = True
            header = self.client.recv(2048).decode()
            if not header:
                self.client.close()
                return None
            size = int(header)
            data = b''
            while size > 0:
                self.client.settimeout(5)
                content = self.client.recv(1024 * 8 * 1024)  # 接收缓冲区最大8M
                data += content
                size -= len(content)
                # Python中recv()是阻塞的，只有连接断开或异常时，接收到的是b''空字节类型，因此需要判断这种情况就断开连接。
                if content == b'':
                    is_conn = False
                    break
            else:
                return pickle.loads(data)
            if not is_conn:
                print('The client is disconnected...')
                return None
        except Exception as e:
            print('Error:', e)
            return None


class Client:

    def __init__(self, sever_host: str, sever_port: int):
        self.socket = socket.socket()
        self.socket.connect((sever_host, sever_port))

    def send(self, obj: object) -> bool:
        """
        :param obj: 需要发送的对象
        :return: 成功返回True，失败返回False
        """
        try:
            data = pickle.dumps(obj)
            size = len(data)
            self.socket.send(str(size).encode())  # 报头
            self.socket.send(data)
            return True
        except Exception as e:
            print('Error:', e)
            return False

    def recv(self):
        """
        :return: 成功返回接收到的对象并反序列化，失败返回None
        """
        try:
            is_conn = True
            header = self.socket.recv(2048).decode()
            if not header:
                self.socket.close()
                return None
            size = int(header)
            data = b''
            while size > 0:
                self.socket.settimeout(5)
                content = self.socket.recv(1024 * 8 * 1024)  # 接收缓冲区最大8M
                data += content
                size -= len(content)
                # Python中recv()是阻塞的，只有连接断开或异常时，接收到的是b''空字节类型，因此需要判断这种情况就断开连接。
                if content == b'':
                    is_conn = False
                    break
            else:
                return pickle.loads(data)
            if not is_conn:
                print('The client is disconnected...')
                return None
        except Exception as e:
            print('Error:', e)
            return None
