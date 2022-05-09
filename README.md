# Image Stitcher

## Requirements
Conda is reccomended to keep the correct version of the dependencies but not necessary.
- Python 3.9+
- numpy
- scipy
- CV2
- tqdm
- skimage
- (optional) tomark to convert results to markdown

## Running the code
To run the model with the default parameters, run the following command:
```
python stitch-imgs.py left_img.jpg right_img.jpg
```
To show help, show a description of each parameters and overwrite them, run the following command:
```
python stitch-imgs.py -h
```