import multiprocessing
import importlib

def pipe_process(pipe, module_name):
    lib = importlib.import_module(module_name)
    func = getattr(lib, module_name)
    pipe.send('Okay :-)')
    while 1:
        data = pipe.recv()
        try:
            poem = func(data)
            pipe.send(poem)
        except Exception as e:
            pipe.send(e)

def create(module_name):
    pipes = multiprocessing.Pipe() # W/R
    proc = multiprocessing.Process(target=pipe_process, args=(pipes[1], module_name))
    proc.start()
    # wait to start
    print (pipes[0].recv())

    def handle(img_feature):
        pipes[0].send(img_feature)
        msg = pipes[0].recv()
        if isinstance(msg, Exception):
            raise msg
        return msg
    return handle

if __name__ == '__main__':
    create('generate_poem')
