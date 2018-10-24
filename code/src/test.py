import nn_process
import time
import sys

print ('Loading Extract Feature Module...')
extract_feature = nn_process.create('extract_feature')
print ('Loading Generate Poem Module...')
generate_poem = nn_process.create('generate_poem')

# default path to an image
DEFAULT_PATH = '../images/test.jpg'

if sys.version_info[0] >= 3:
    raw_input = input

def get_poem(image_file):
    """Generate a poem from the image whose filename is `image_file`

    Parameters
    ----------
    image_file : str
        Path to the input image

    Returns
    -------
    str
        Generated Poem
    """
    img_feature = extract_feature(image_file)
    return generate_poem(img_feature)

if __name__ == '__main__':
    while 1:
        try:
            s = raw_input("Please input the path to an image [default='%s']: " % DEFAULT_PATH)
            if not s:
                # s is empty, and use the DEFAULT_PATH
                s = DEFAULT_PATH
            tic = time.time()
            print ('\n' + get_poem(s)[0].replace('\n', '\n') + '\n')
            print ("Cost Time: %f" % (time.time() - tic))
        except Exception as e:
            print (e)
