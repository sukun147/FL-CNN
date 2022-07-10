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
        socket.setdefaulttimeout(60)
        self.socket = socket.socket()
        self.socket.bind((host, port))
        self.socket.listen(client_num)
        self.client = []  # 客户端套接字
        self.client_flag = {}  # 客户端公钥:客户端套接字
        for i in range(client_num):
            client, address = self.socket.accept()
            self.client.append(client)

    def send(self, obj: object, flag: str = '') -> bool:
        """
        向所有或指定客户端发送对象\n
        :param obj: 需要发送的对象
        :param flag: 用于指定客户端，默认为空时进行群发，给定flag查找指定客户端进行发送，查找失败抛出异常
        :return: 成功返回True，失败返回False
        """
        try:
            if flag:
                client = [self.client_flag[flag]]
            else:
                client = self.client
            data = pickle.dumps(obj)
            header = json.dumps({'size': len(data)}).encode()
            for c in client:
                c.sendall(struct.pack('i', len(header)))  # 报头长度
                c.sendall(header)  # 报头
                c.sendall(data)  # 数据
            return True
        except Exception as e:
            print('Error:', e)
            return False

    def recv(self, flag: str = ''):
        """
        接收所有或指定客户端发送的对象\n
        :param flag: 用于指定客户端，默认为空时接收所有客户端，给定flag查找指定客户端接收，查找失败抛出异常
        :return: 成功返回对象，客户端套接字二元组的迭代器，失败返回None
        """
        try:
            if flag:
                client = [self.client_flag[flag]]
            else:
                client = self.client
            for c in client:
                is_conn = True
                # 接收报头长度
                header_struct = c.recv(4)
                header_len = struct.unpack('i', header_struct)[0]
                # 接收报头
                header = c.recv(header_len)
                if not header:
                    c.close()
                    return None
                size = json.loads(header.decode())['size']
                data = b''
                while size > 0:
                    content = c.recv(1024 * 8 * 1024)  # 接收缓冲区最大8M
                    data += content
                    size -= len(content)
                    # Python中recv()是阻塞的，只有连接断开或异常时，接收到的是b''空字节类型，因此需要判断这种情况就断开连接。
                    if content == b'':
                        is_conn = False
                        break
                else:
                    yield pickle.loads(data), c
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
            socket.setdefaulttimeout(60)
            self.socket = socket.socket()
            self.socket.connect((sever_host, sever_port))
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
            self.socket.sendall(struct.pack('i', len(header)))  # 报头长度
            self.socket.sendall(header)  # 报头
            self.socket.sendall(data)  # 数据
            return True
        except Exception as e:
            print('Error:', e)
            return False

    def recv(self):
        """
        接收所有客户端发送的对象\n
        :return: 成功返回接收对象，失败返回None
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
                print('The server is disconnected...')
                return None
        except Exception as e:
            print('Error:', e)
            return None
