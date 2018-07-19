# img2poem

Environment: Tensorflow1.6

# Data

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

```
[
	{
		"poem": str,
		"id": int
	},
	...
]
```