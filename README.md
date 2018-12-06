# seam-carving

<img src="https://github.com/andrewdcampbell/seam-carving/blob/master/demos/visuals/lake_shrink.gif" width="900">

A fast Python implementation of [Seam Carving for Content-Aware Image Resizing ](https://inst.eecs.berkeley.edu/~cs194-26/fa18/hw/proj4-seamcarving/imret.pdf) (2007), including the improved energy algorithm described in [Improved Seam Carving for Video Retargeting](http://www.eng.tau.ac.il/~avidan/papers/vidret.pdf) (2008).

## Requirements
* OpenCV
* scipy
* numba
* numpy

## Usage
```
python seam_carving.py (-resize | -remove) -im <IM_PATH> -out <OUTPUT_IM_NAME> 
                       [-mask <MASK_PATH>] [-rmask <REMOVAL_MASK_PATH>] [-dy <DY>] [-dx <DX>] 
                       [-vis] [-hremove] [-backward_energy]
```

The program is run via the command-line. There are two modes of operations: `resize` or `remove`. The former is for resizing an image vertically or horizontally and the latter is for removing an object as specified by a mask.

For both modes:
* `-im`: The path to the image to be processed.
* `-out`: The name for the output image.
* `-mask`: (Optional) The path to the protective mask. The mask should be binary and have the same size as the input image. White areas represent regions where no seams should be carved (e.g. faces).
* `-vis`: If present, display a window while the algorithm runs showing the seams as they are removed.
* `-backward_energy`: If present, use the backward energy function (i.e. gradient magnitude) instead of the forward energy function (default).

For resizing:
* `-dy`: Number of horizontal seams to add (if positive) or subtract (if negative). Default is 0.
* `-dx`: Number of vertical seams to add (if positive) or subtract (if negative). Default is 0.

For object removal:
* `-rmask`: The path to the removal mask. The mask should be binary and have the same size as the input image. White areas represent regions to be removed.
* `-hremove`: If present, perform seam removal horizontally rather than vertically. This will be more appropriate in certain contexts.


#### Additional Parameters
There are some additional constants defined at the top of the code `seam_carving.py` that may be modified.
* The code currently downsizes any images with width larger than 500 pixels to 500 pixels for super fast carving. To change this, change the value of `DOWNSIZE_WIDTH`, or set `SHOULD_DOWNSIZE` to `False` to disable downsizing completely.
* Seams are visualized as a bluish-white color; change the color by changing the `SEAM_COLOR` array (in BGR format).


## Example Results

The input image is on the left and the result of the algorithm is on the right.

### Vertical Seam Removal

<img src="https://github.com/andrewdcampbell/seam-carving/blob/master/demos/castle.jpg" height="342"> <img src="https://github.com/andrewdcampbell/seam-carving/blob/master/demos/castle_shrink.jpg" height="342">

### Horizontal Seam Removal

<img src="https://github.com/andrewdcampbell/seam-carving/blob/master/demos/museum.jpg" width="415"> <img src="https://github.com/andrewdcampbell/seam-carving/blob/master/demos/museum_shrink.jpg" width="415">

### Seam Removal with Protective Masks

<img src="https://github.com/andrewdcampbell/seam-carving/blob/master/demos/ratatouille.jpg" height="313"> <img src="https://github.com/andrewdcampbell/seam-carving/blob/master/demos/ratatouille_resize.jpg" height="313">

### Seam Insertion

<img src="https://github.com/andrewdcampbell/seam-carving/blob/master/demos/shore.jpg" height="460"> <img src="https://github.com/andrewdcampbell/seam-carving/blob/master/demos/shore_backward_energy_expand.jpg" height="460">

### Object Removal with Protective Masks

<img src="https://github.com/andrewdcampbell/seam-carving/blob/master/demos/gotcast.jpg" height="294"> <img src="https://github.com/andrewdcampbell/seam-carving/blob/master/demos/kit_remove.jpg" height="294">

### Object Removal with Seam Insertion

<img src="https://github.com/andrewdcampbell/seam-carving/blob/master/demos/eiffel.jpg" height="230"> <img src="https://github.com/andrewdcampbell/seam-carving/blob/master/demos/eiffel_forward_removal.jpg" height="230">

Animated gif:

<img src="https://github.com/andrewdcampbell/seam-carving/blob/master/demos/visuals/eiffel_remove.gif" width="700">

## Comparison between Energy Functions

*In general*, forward energy gives better results than backward energy. The result of resizing using backward energy (left) and forward energy (right) is shown below.

<img src="https://github.com/andrewdcampbell/seam-carving/blob/master/demos/plane_shrink_backward.jpg" width="400"> <img src="https://github.com/andrewdcampbell/seam-carving/blob/master/demos/plane_shrink_forward.jpg" width="400">

<img src="https://github.com/andrewdcampbell/seam-carving/blob/master/demos/bench_backward_energy.jpg" width="400"> <img src="https://github.com/andrewdcampbell/seam-carving/blob/master/demos/bench_forward_energy.jpg" width="400">

---
For more information on how the algorithm works, see my [blog post](https://andrewdcampbell.github.io/seam-carving). 

## Acknowledgements
Many parts of the code are adapted/optimized versions of functionality from other implementations:
* https://github.com/axu2/improved-seam-carving
* https://github.com/vivianhylee/seam-carving
* https://karthikkaranth.me/blog/implementing-seam-carving-with-python/
