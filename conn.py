import socket
import pickle
import json
import os

tmp_file = 'tmp.txt'


class Sever:

    def __init__(self, host: str, port: int, client_num: int):
        self.socket = socket.socket()
        self.socket.bind((host, port))
        self.socket.listen(client_num)

    def send(self, obj: object) -> bool:
        """
        :param obj: 需要发送的对象
        :return: 成功返回True，失败返回False
        """
        try:
            with open(tmp_file, 'wb') as f:
                pickle.dump(obj, f)
                file_size = os.path.getsize(tmp_file)
            header = {"file_name": tmp_file, "file_size": file_size}  # 自定义报头
            self.socket.send(json.dumps(header).encode())  # 序列化报头
            file_seek = int(self.socket.recv(100).decode())
            if file_seek == file_size:
                print('The file already exists on the client. Exit the transfer...')
            else:
                new_size = file_size - file_seek
                with open(tmp_file, 'rb') as f:
                    f.seek(file_seek)  # 移动文件指针到文件的指定位置
                    while new_size > 0:
                        content = f.read(1024 * 8 * 1024)  # 接收缓冲区最大为8M，故每次读取8M来发送
                        self.socket.send(content)
                        new_size -= len(content)
                    else:
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
            data = self.socket.recv(2048)
            if not data:
                self.socket.close()
                return None
            header = json.loads(data.decode())
            file_path = header.get('file_name')
            if os.path.exists(file_path):
                file_seek = os.path.getsize(file_path)
            else:
                file_seek = 0
            # 将文件指针发送过去，同时也可以解决粘包
            self.socket.send(str(file_seek).encode())
            if file_seek == header['file_size']:
                print('The file has been transferred. Exit the transfer...')
            else:
                new_size = header['file_size'] - file_seek  # 重新设置需要接收的文件大小
                # 准备接收发来的追加文件内容
                with open(file_path, 'ab') as f:
                    while new_size > 0:
                        self.socket.settimeout(5)
                        content = self.socket.recv(1024 * 8 * 1024)  # 接收缓冲区最大8M
                        f.write(content)
                        new_size -= len(content)
                        # Python中recv()是阻塞的，只有连接断开或异常时，接收到的是b''空字节类型，因此需要判断这种情况就断开连接。
                        if content == b'':
                            is_conn = False
                    else:
                        with open(file_path, 'rb') as ff:
                            return pickle.load(ff)
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
            with open(tmp_file, 'wb') as f:
                pickle.dump(obj, f)
                file_size = os.path.getsize(tmp_file)
            header = {"file_name": tmp_file, "file_size": file_size}
            self.socket.send(json.dumps(header).encode())
            file_seek = int(self.socket.recv(100).decode())
            if file_seek == file_size:
                print('The file already exists on the server. Exit the transfer...')
            else:
                new_size = file_size - file_seek
                with open(tmp_file, 'rb') as f:
                    f.seek(file_seek)
                    while new_size > 0:
                        content = f.read(1024 * 8 * 1024)
                        self.socket.send(content)
                        new_size -= len(content)
                    else:
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
            data = self.socket.recv(2048)
            header = json.loads(data.decode())
            file_path = header.get('file_name')
            if os.path.exists(file_path):
                file_seek = os.path.getsize(file_path)
            else:
                file_seek = 0
            self.socket.send(str(file_seek).encode())
            if file_seek == header['file_size']:
                print('The file has been transferred. Exit the transfer...')
            else:
                new_size = header['file_size'] - file_seek
                with open(file_path, 'ab') as f:
                    while new_size > 0:
                        self.socket.settimeout(5)
                        content = self.socket.recv(1024 * 8 * 1024)
                        f.write(content)
                        new_size -= len(content)
                        if content == b'':
                            is_conn = False
                    else:
                        with open(file_path, 'rb') as ff:
                            return pickle.load(ff)
            if not is_conn:
                print('The server is disconnected...')
                return None
        except Exception as e:
            print('Error:', e)
            return None
