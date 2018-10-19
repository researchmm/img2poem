import caffe
import mxnet as mx

def get_params(caffe_net, caffe_model):
    nn = caffe.Net(caffe_net, caffe_model, caffe.TEST)
    arg_params = dict()
    aux_params = dict()
    for name in nn.params.keys():
        p = nn.params[name] 
        weight = p[0].data
        bias = p[1].data
        arg_params["%s_weight" % (name)] = mx.nd.array(weight) 
        arg_params["%s_bias" % (name)] = mx.nd.array(bias) 
    return arg_params, aux_params

def save_params(filename, arg_params, aux_params):
    data = {}
    for name, value in arg_params.items():
        data["arg:%s" % name] = value
    for name, value in aux_params.items():
        data["aux:%s" % name] = value
    mx.nd.save(filename, data)

if __name__ == "__main__":
    CAFFE_NET = './vgg/deploy.prototxt' 
    CAFFE_MODEL = './vgg/VGG_ILSVRC_16_layers.caffemodel'
    arg_params, aux_params = get_params(CAFFE_NET, CAFFE_MODEL)
    save_params("vgg16.params", arg_params, aux_params)
