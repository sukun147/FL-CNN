def model_average(param_list):

    '''模型平均

    :param param_list: 客户端的模型参数列表，其中每个元素仍为列表，单个列表元素为单个客户端的参数

    :return: 返回一个包含已完成平均操作的模型参数列表

    '''

    new_params = param_list[0]

    for i in range(1, len(param_list)):
        for j in range(len(new_params)):
            new_params[j] += param_list[i][j]

    for i in range(len(new_params)):
        new_params[i] /= len(param_list)

    return new_params

def update_model_params(model, params: list, grads: list, lr=0.1):

    '''更新模型参数

    :param model: 用于完成图像识别的模型，当前使用CNN（此处为单个客户端的model）

    :param params: 由模型平均返回的新参数（直接导入模型)
    :param grads: 由梯度平均返回的各客户端的参数平均梯度(根据梯度下降公式更新模型参数)
    以上两者在传入函数时，应有且仅有一项为空列表，分别对应模型平均与梯度平均的结果

    :param lr: 模型的学习率(learning rate)，默认值取0.1

    :return: 无返回值，模型在函数内部完成更新

    '''

    if len(params) != 0:

        for new_parma, param in zip(params, model.parameters()):
            param.data = new_parma.to(param.device)

    else:
        with torch.no_grad():

            for grad, param in zip(grads, model.parameters()):
                param.data -= lr * grad.to(param.device)
