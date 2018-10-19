import mxnet as mx
import numpy as np
import cv2
from symbol_vgg import VGG
from collections import namedtuple

MEAN_COLOR = np.array([110.474, 118.574, 123.955]).reshape((1, 3, 1, 1)) # BGR 

def load_checkpoint(prefix, epoch):
    """
    Load model checkpoint from file.
    :param prefix: Prefix of model name.
    :param epoch: Epoch number of model we would like to load.
    :return: (arg_params, aux_params)
    arg_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's weights.
    aux_params : dict of str to NDArray
        Model parameter, dict of name to NDArray of net's auxiliary states.
    """
    save_dict = mx.nd.load('%s-%04d.params' % (prefix, epoch))
    arg_params = {}
    aux_params = {}
    for k, v in save_dict.items():
        tp, name = k.split(':', 1)
        if tp == 'arg':
            arg_params[name] = v
        if tp == 'aux':
            aux_params[name] = v
    return arg_params, aux_params

vgg = VGG()
sym = vgg.get_symbol(num_classes = 1000, 
          blocks = [(2, 64),
                    (2, 128),
                    (3, 256), 
                    (3, 512),
                    (3, 512)])

ctx = mx.cpu(0)

mod = mx.module.Module(
        context = ctx,
        symbol = sym,
        data_names = ("data", ),
        label_names = ("softmax_label", )
)

mod.bind(data_shapes = [("data", (1, 3, 224, 224))])

mod.load_params("./vgg16.params")

Batch = namedtuple('Batch', ['data'])

fn = "mobula.jpg"

img = cv2.imread(fn) # BGR
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

rows, cols, ts = img.shape
if rows < cols:
    r = 224
    c = cols * 224 // rows
    o = (c - 224) // 2
    img = cv2.resize(img, (c, r))
    if o > 0:
        img = img[:, o:-o, :]
else:
    c = 224
    r = rows * 224 // cols
    o = (r - 224) // 2
    img = cv2.resize(img, (c, r))
    if o > 0:
        img = img[o:-o, :, :]

img = img.transpose((2,0,1)).reshape((1, 3, 224, 224))

img = img.astype(np.float32)

img = img - MEAN_COLOR # N,C,H,W

bdata = (mx.nd.array(img, ctx))

mod.forward(Batch([bdata]), is_train = False)
outputs = mod.get_outputs()[0].asnumpy()

p = outputs[0]
pred = [(i, p) for i, p in enumerate(p)]
pred.sort(key = lambda x : x[1], reverse = True)

fin = open("./inet.txt")
cls = [line.strip() for line in fin]
assert len(cls) == 1000, cls

for t in range(5):
    u = pred[t]
    i = u[0]
    print (cls[i], u[1])
