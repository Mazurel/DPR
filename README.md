# DPR wrapper

This is a simple wrapper around: <b>Deep Single-Image Portrait Relighting</b> [[Project Page]](http://zhhoper.github.io/dpr.html) <br>
Hao Zhou, Sunil Hadap, Kalyan Sunkavalli, David W. Jacobs. In ICCV, 2019

## Using as a module

Currently this project cannot be used as normal Python module.
To use it, you will need to clone it directly into your source code.

I hope to fix it soon.

## Usage

```python
from DPR import DPR_512

IMAGE = "test.png"

dpr = DPR_512()
input_image, output_image = dpr.random_relighten(IMAGE)

# Now images represented as numpy arrays are in proper variables.
# For example, you can use them like so:

form matplotlib import pyplot as plt
plt.figure("Before")
plt.imshow(input_image)
plt.figure("After")
plt.imshow(output_image)
plt.show()
```

Examples can also be found in:

- `transform_512.py`
- `transform_1024.py`
