import socket
import pickle
import json
import struct


class Server:

    def __init__(self, host: str, port: int, client_num: int):
        """
        :param host: 服务器地址，可填''即本机
        :param port: 服务器端口
        :param client_num: 客户端数量
        """
        self.socket = socket.socket()
        self.socket.settimeout(60)
        self.socket.bind((host, port))
        self.socket.listen(client_num)
        self.client = None

    def accept(self):
        """
        :return: 连接客户端成功返回True，失败返回False
        """
        client, port = self.socket.accept()
        self.client = client
        self.socket.settimeout(None)
        return True if client else False

    def send(self, obj: object) -> bool:
        """
        :param obj: 需要发送的对象
        :return: 成功返回True，失败返回False
        """
        try:
            data = pickle.dumps(obj)
            header = json.dumps({'size': len(data)}).encode()
            self.client.send(struct.pack('i', len(header)))  # 报头长度
            self.client.send(header)  # 报头
            self.client.send(data)  # 数据
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
            # 接收报头长度
            header_struct = self.client.recv(4)
            header_len = struct.unpack('i', header_struct)[0]
            # 接收报头
            header = self.client.recv(header_len)
            if not header:
                self.client.close()
                return None
            size = json.loads(header.decode())['size']
            data = b''
            while size > 0:
                self.client.settimeout(60)
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
        """
        :param sever_host: 服务端地址
        :param sever_port: 服务端端口
        """
        try:
            self.socket = socket.socket()
            self.socket.settimeout(60)
            self.socket.connect((sever_host, sever_port))
            self.socket.settimeout(None)
        except ConnectionRefusedError:
            print('please start the server first...')
            return

    def send(self, obj: object) -> bool:
        """
        :param obj: 需要发送的对象
        :return: 成功返回True，失败返回False
        """
        try:
            data = pickle.dumps(obj)
            header = json.dumps({'size': len(data)}).encode()
            self.socket.send(struct.pack('i', len(header)))  # 报头长度
            self.socket.send(header)  # 报头
            self.socket.send(data)  # 数据
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
            # 接收报头长度
            header_struct = self.socket.recv(4)
            header_len = struct.unpack('i', header_struct)[0]
            # 接收报头
            header = self.socket.recv(header_len)
            if not header:
                self.socket.close()
                return None
            size = json.loads(header.decode())['size']
            data = b''
            while size > 0:
                self.socket.settimeout(60)
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
