import mxnet as mx

class VGG:
    def vgg_block(self, x, num_convs, channels, block_id):
        for i in range(num_convs):
            x = mx.sym.Convolution(data = x, num_filter = channels, kernel = (3, 3), stride = (1, 1), pad = (1, 1), no_bias = False, name = 'conv%d_%d' % (block_id, i + 1))
            x = mx.sym.Activation(data = x, act_type = 'relu', name = 'relu%d_%d' % (block_id, i + 1))
        x = mx.sym.Pooling(data = x, kernel = (2, 2), pool_type = 'max', stride = (2, 2), name = 'pool%d' % block_id)
        return x

    def get_symbol(self, num_classes, blocks, dropout = 0.5):
        '''
        Return VGG symbol
        Parameters
        ----------
        num_classes: int
            The number of classes
        blocks : list
            The list of convolutional layers (num_convs, channels)
        '''
        x = mx.sym.Variable(name = 'data')

        for i, b in enumerate(blocks):
            x = self.vgg_block(x, num_convs = b[0], channels = b[1], block_id = i + 1)

        x = mx.sym.flatten(x)
        x = mx.sym.FullyConnected(data = x, num_hidden = 4096, name = 'fc6')
        x = mx.sym.Activation(data = x, act_type = 'relu', name = 'relu6')
        x = mx.sym.Dropout(data = x, p = dropout, name = 'dropout6')

        x = mx.sym.FullyConnected(data = x, num_hidden = 4096, name = 'fc7')
        x = mx.sym.Activation(data = x, act_type = 'relu', name = 'relu7')
        x = mx.sym.Dropout(data = x, p = dropout, name = 'dropout7')

        x = mx.sym.FullyConnected(data = x, num_hidden = num_classes, name = 'fc8')
        return mx.symbol.SoftmaxOutput(data = x, name = 'softmax')
