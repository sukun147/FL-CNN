import utils
import conn

...
epoch = ...
client = conn.Client(..., ...)
model = ...


def train():
    ...


for i in range(epoch):
    ...
    params = train()
    # 向服务端发送参数
    client.send(params)
    # 接收服务端发送的参数
    avg_params = client.recv()
    # 更新参数
    utils.update_model_params(model)
    ...
