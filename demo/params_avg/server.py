from multiprocessing import context
import sys
sys.path.append(".\\.\\") # 使其能够调用上上级目录的文件
import utils
import conn
import tenseal as ts
import numpy as np
import warnings
warnings.filterwarnings("ignore")
epoch = 100
client_num = 2
host ='0.0.0.0'
port =666
server = conn.Server(host, port, client_num)
server.client_flag = {k: v for k, v in server.recv()}
server.send(client_num,flag='client1')
server_context=server.recv(flag='client1').__next__()[0]
server.send('1',flag='client1')
server.send(server_context,flag='client2')
context=ts.context_from(server_context)
for i in range(epoch):
    params_list = []
    # 接收客户端发送过来的参数
    for j in server.recv():
        params_list.append(j[0])
    print(len(params_list))
    new_params=params_list[0]
    for i in range(len(params_list)):
        for j in range(len(new_params)):
            params_list[i][j]=ts.ckks_vector_from(context, params_list[i][j])
    # 参数聚合
    avg_params = utils.model_average(params_list)
    for i in range(len(avg_params)):
        avg_params[i]=avg_params[i].serialize()
    # 分发参数
    server.send(avg_params,flag='client1')
    avg_params=server.recv(flag='client1').__next__()[0]
    server.send(avg_params,flag='client2')
