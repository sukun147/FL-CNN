import conn

client_num = 2
server = conn.Server('', 5556, client_num)  # 不填写ip，默认为本机

if __name__ == '__main__':
    while True:
        if server.accept():
            print(server.recv())
