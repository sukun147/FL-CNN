import utils
import conn

epoch = ...
client_num = 2
sever = conn.Sever(..., ..., client_num)

for i in range(epoch):
    params_list = []
    # 接收客户端发送过来的参数
    for j in range(client_num):
        params_list.append(sever.recv())

    # 参数聚合
    avg_params = utils.model_average(params_list)

    # 分发参数
    for j in range(client_num):
        sever.send(avg_params)
