import conn

client_num = 2
server = conn.Server('', 5556, client_num)  # 不填写ip，默认为本机

if __name__ == '__main__':
    # 首先接收到客户端发送自己的标识并存起来
    server.client_flag = {k: v for k, v in server.recv()}
    server.send('1')
    for i in server.recv():
        print(i[0])
    server.send('hello client1', flag='client1')
    server.send('hello client2', flag='client2')
