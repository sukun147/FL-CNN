import conn

client_num = 1
server = conn.Server('', 5556, client_num)  # 不填写ip，默认为本机

if __name__ == '__main__':
    for i in server.recv():
        print(i)
    server.send('hello')
