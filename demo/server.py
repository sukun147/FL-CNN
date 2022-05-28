import utils
import conn

epoch = 100
client_num = 2
host = ''
port = 666
server = conn.Server(host, port, client_num)

for i in range(epoch):
    params_list = []
    # 接收客户端发送过来的参数
    for i in server.recv():
        params_list.append(i)

    # 参数聚合
    avg_params = utils.model_average(params_list)

    # 分发参数
    server.send(avg_params)
