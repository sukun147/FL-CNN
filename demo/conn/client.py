import conn

client = conn.Client('192.168.1.106', 5555)  # 服务器ip，服务器端口

if __name__ == '__main__':
    client.send('Hello World' * 1000)
