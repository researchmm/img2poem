# img2poem


## Environment:
(It is recommended to install the dependencies under Conda environment.)  
python2.7  
tensorflow1.6  
mxnet  
opencv  
scikit-image  
tqdm  
colorama  
flask  

## Model
Please download models from https://1drv.ms/u/s!AkLgJBAHL_VFgSyyfpeGyGFZux56 and put it under "code/".

## Data

Both datasets are formatted in JSON files.

MultiM-Poem.json: image and poem pairs

```json
[
	{
		"poem": str,
		"image_url": str,
		"id": int
	},
	...
]
```

UniM-Poem.json: poem corpus

```json
[
	{
		"poem": str,
		"id": int
	},
	...
]
```
