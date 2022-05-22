import conn

client_num = 2
server = conn.Server('', 5555, client_num)  # 不填写ip，默认为本机

if __name__ == '__main__':
    print(server.recv())
