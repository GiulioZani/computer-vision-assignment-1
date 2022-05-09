# Assigment 1 Computer Vision
## Giulio Zani

## Step 1
See file `corner_harris.py`

## Step 2
See the function `stitch` in file `stitch-imgs.py`

## Selecting Descriptors
After the descriptors are extracted, they are selected based on their distance between all of the descriptors of the other image. This will yield a NxM matrix, where N is the number of descriptors found in the left image and M in the right. The minimum for each row is found and then the descriptors are selected usign a threshold that is *relative*. So if `t` is the threshold, and `k` is the number of descriptors:

$$ k = tN $$

As mentioned in the assigment description, there are two ways to compute the distance between two descriptors: *eucledian* and *normalized correlation*.
The results of both techniques are reported in the following subsections.
These results are based on the sensitivity analysis described in its corresponding section.

### Normalized Correlation

### Hyperparameters
| ransac iters | ransac sample size | ransac distance threshold | ransac accepted threshold | distance method | top matches |
|-----|-----|-----|-----|-----|-----|
| 10000 | 0.2 | 0.1 | 0.01 | correlation | 0.04 |

### Results
| accuracy | error |
|-----|-----|
| 0.04348 | 0.0737 |

### Euclidian Distance
### Hyperparameters
| ransac iters | ransac sample size | ransac distance threshold | ransac accepted threshold | distance method | top matches |
|-----|-----|-----|-----|-----|-----|
| 10000 | 0.2 | 0.1 | 0.01 | euclidean | 0.04 |

### Results
| accuracy | error |
|-----|-----|
| 0.1087 | 0.05039 |

## Ransac 
While implementing the Ransac algorithm, I have tried as much as possible to use *relative* thresholds. This is because the number of descriptors and the sample size are not known in advance and thus using relative thresholds makes the algorithm more robust.

### Hyperparameters
| left_img_path | right_img_path | ransac_iters | ransac_sample_size | ransac_distance_threshold | ransac_accepted_threshold | save_path | distance_method | top_matches |
|-----|-----|-----|-----|-----|-----|-----|-----|-----|
| ./sample_images/2l.jpg | ./sample_images/2r.jpg | 10000 | 0.2 | 0.1 | 0.01 | ./out_imgs | euclidean | 0.04 |

### Results
| accuracy | error |
|-----|-----|
| 0.15217 | 0.04955 |

## Sensitivity Analisys 
As requested, I have performed a sensitivity analysis on the Ransac algorithm using the ratio number of inliers/number of outliers.


## Stitching images

## Experimenting with Hyperparameters