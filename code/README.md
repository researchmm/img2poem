images: test images

model: pretrained models, including image feature extraction model (object.params, scene.params, Sentiment.params) and poem generation model (ckpt/)

src: code for testing
-- test.py
to test how much time it cost to generate a poem for an image

get_poem(image_file): 
image_file: path to the input image
return: generated poem