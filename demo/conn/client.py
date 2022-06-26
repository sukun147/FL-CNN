import conn

client = conn.Client('192.168.31.98', 5556)  # 服务器ip，服务器端口

if __name__ == '__main__':
    client.send('client1')
    if client.recv() == '1':
        client.send('hello world1')
        print(client.recv())
