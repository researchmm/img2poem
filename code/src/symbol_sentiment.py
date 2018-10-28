import mxnet as mx

def LRN(x):
    lrn_kwargs = {
            'knorm': 1,
            'alpha': 1e-4,
            'beta' : 0.75,
            'nsize': 5,
    }
    x = mx.sym.LRN(data = x, **lrn_kwargs)
    return x

def get_sym():

    x = mx.sym.Variable(name = 'data')
    x = mx.sym.Convolution(data = x, num_filter = 96, kernel = (11, 11), stride = (4, 4), name = 'conv1_A')
    x = mx.sym.Activation(data = x, act_type = 'relu')
    x = mx.sym.Pooling(data = x, kernel = (3, 3), pool_type = 'max', stride = (2, 2), pooling_convention = 'full')
    x = LRN(x) # 27 * 27

    x = mx.sym.Convolution(data = x, num_filter = 256, kernel = (5, 5), pad = (2, 2), num_group = 2, name = 'conv2_A')
    x = mx.sym.Activation(data = x, act_type = 'relu')
    x = mx.sym.Pooling(data = x, kernel = (3, 3), pool_type = 'max', stride = (2, 2), pooling_convention = 'full')
    x = LRN(x) # 13 * 13

    x = mx.sym.Convolution(data = x, num_filter = 384, kernel = (3, 3), pad = (1, 1), name = 'conv3_A')
    x = mx.sym.Activation(data = x, act_type = 'relu')

    x = mx.sym.Convolution(data = x, num_filter = 384, kernel = (3, 3), pad = (1, 1), num_group = 2, name = 'conv4_A')
    x = mx.sym.Activation(data = x, act_type = 'relu')

    x = mx.sym.Convolution(data = x, num_filter = 256, kernel = (3, 3), pad = (1, 1), num_group = 2, name = 'conv5_A')
    x = mx.sym.Activation(data = x, act_type = 'relu')
    x = mx.sym.Pooling(data = x, kernel = (3, 3), pool_type = 'max', stride = (2, 2), pooling_convention = 'full')

    x = mx.sym.flatten(x)
    x = mx.sym.FullyConnected(data = x, num_hidden = 4096, name = 'fc6_A')
    x = mx.sym.Activation(data = x, act_type = 'relu', name = 'relu6_A')

    x = mx.sym.FullyConnected(data = x, num_hidden = 4096, name = 'fc7_A')
    x = mx.sym.Activation(data = x, act_type = 'relu', name = 'relu7_A')
    return x
