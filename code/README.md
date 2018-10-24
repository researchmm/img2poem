# img2poem

images/: test images

model/: pretrained models, including image feature extraction model (object.params, scene.params, Sentiment.params) and poem generation model (ckpt/)

src/: code for testing

- test.py

To test how much time it cost to generate a poem for an image

```python
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
```
