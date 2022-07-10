import conn

client = conn.Client('127.0.0.1', 5556)  # 服务器ip，服务器端口

if __name__ == '__main__':
    client.send('client1')
    if client.recv() == '1':
        client.send('hello world1')
        print(client.recv())
        client.send('end')
