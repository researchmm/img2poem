import extract_feature
import generate_poem
import time

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
    img_feature = extract_feature.get_feature(image_file)
    return generate_poem.generate(img_feature)

if __name__ == '__main__':
    while 1:
        try:
            s = raw_input()
            tic = time.time()
            print (get_poem(s))
            print ("Cost Time: %f" % (time.time() - tic))
        except Exception as e:
            print (e)

    for _ in range(100):
        tic = time.time()
        print (get_poem('../images/test.jpg'))
        print ("Cost Time: %f" % (time.time() - tic))
