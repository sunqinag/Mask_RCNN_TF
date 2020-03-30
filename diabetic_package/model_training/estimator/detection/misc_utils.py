# coding: utf-8
import numpy as np
import tensorflow as tf

def load_weights(var_list, weights_file):
    """
    加载并转换预训练的权重参数
    参数:
        var_list: 网络的参数列表.
        weights_file: 二值文件的名字.
    """
    with open(weights_file, "rb") as fp:
        np.fromfile(fp, dtype=np.int32, count=5)
        weights = np.fromfile(fp, dtype=np.float32)

    ptr = 0
    i = 0
    assign_ops = []
    while i < len(var_list) - 1:
        var1 = var_list[i]
        var2 = var_list[i + 1]

        if 'conv2d' in var1.name.split('/')[-2]:
            if 'batch_normalization' in var2.name.split('/')[-2]:
                gamma, beta, mean, var = var_list[i + 1:i + 5]
                batch_norm_vars = [beta, gamma, mean, var]
                for var in batch_norm_vars:
                    shape = var.shape.as_list()
                    num_params = np.prod(shape)
                    var_weights = weights[ptr:ptr + num_params].reshape(shape)
                    ptr += num_params
                    assign_ops.append(tf.assign(var, var_weights, validate_shape=True))
                i += 4
            elif 'conv2d' in var2.name.split('/')[-2]:
                var2_shape = var2.shape.as_list()
                var2_params = np.prod(var2_shape)
                bias_weights = weights[ptr:ptr +
                                           var2_params].reshape(var2_shape)
                ptr += var2_params
                assign_ops.append(tf.assign(var2, bias_weights, validate_shape=True))
                i += 1
            shape = var1.shape.as_list()
            num_params = np.prod(shape)

            var_weights = weights[ptr:ptr + num_params].reshape(
                (shape[3], shape[2], shape[0], shape[1]))
            var_weights = np.transpose(var_weights, (2, 3, 1, 0))
            ptr += num_params
            assign_ops.append(
                tf.assign(var1, var_weights, validate_shape=True))
            i += 1


    return assign_ops


