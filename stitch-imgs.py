import cv2
from cv2 import KeyPoint
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser
from scipy.spatial.distance import cdist
from tqdm import tqdm
import json
import os
from skimage.feature import corner_harris, corner_subpix, corner_peaks

tomark_installed = True
try:
    from tomark import Tomark
except ImportError as e:
    tomark_installed = False


def compute_corner_harris(image: np.ndarray, save_path: str):

    _, ax = plt.subplots() # create a figure and an axes
    h_corners = corner_harris(image, k=0.02) # find Harris corners
    peaks = corner_peaks(h_corners, min_distance=15, threshold_rel=0.01) # find peaks
    sub_pix = corner_subpix(h_corners, peaks, window_size=10) # refine corners+
    ax.plot(
        peaks[:, 1],
        peaks[:, 0],
        color="cyan",
        marker="o",
        linestyle="None",
        markersize=6,
    )
    ax.plot(sub_pix[:, 1], sub_pix[:, 0], "+r", markersize=15)
    ax.imshow(image, cmap=plt.cm.gray) # plot the image
    plt.savefig(os.path.join(save_path, "corner_harris.png"))
    plt.clf()
    plt.close()

def draw_keypoints_on_img(
    gray_img: np.ndarray, rgb_img: np.ndarray, keypoints: list[KeyPoint]
):
    return cv2.drawKeypoints( # draw keypoints on image
        gray_img,
        keypoints,
        rgb_img.copy(),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
    )


def read_img(path: str):
    img = cv2.imread(path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_gray, img_rgb

def normalize(d1: np.ndarray) -> np.ndarray:
    max_val = np.max(d1) # get the max value
    min_val = np.min(d1)
    
    norm_d1 = (d1 - min_val) / (max_val - min_val) # normalize
    # normalize the descriptors
    assert (norm_d1 <= 1).all() and (
        norm_d1 >= 0
    ).all(), "Normalization failed"
    return norm_d1


def get_matches(
    left_descriptors: np.ndarray,
    right_descriptors: np.ndarray,
    left_keypoints: list[KeyPoint],
    right_keypoints: list[KeyPoint],
    top_matches: float = 0.05,
    method="correlation",
):
    left_keypoints = np.array(left_keypoints)
    right_keypoints = np.array(right_keypoints)
    max_desc_value = max(np.max(left_descriptors), np.max(right_descriptors))
    min_desc_value = min(np.min(left_descriptors), np.min(right_descriptors))
    left_descriptors = (left_descriptors - min_desc_value) / (
        max_desc_value - min_desc_value
    ) # normalize the descriptors
    right_descriptors = (right_descriptors - min_desc_value) / (
        max_desc_value - min_desc_value
    ) # normalize the descriptors
    # computes the distance between the normalized descriptors using cdist
    distances = cdist(left_descriptors, right_descriptors, method)
    sorting_indices = np.stack(
        (np.arange(len(distances)), np.argmin(distances, axis=1)), axis=1
    ) # get the indices of the sorted distances
    row_sorting = np.argsort(
        distances[sorting_indices[:, 0], sorting_indices[:, 1]]
    ) # sort the rows  
    max_matches = int(len(sorting_indices) * top_matches)
    sorted_keypoint_indices = sorting_indices[row_sorting][:max_matches]
    selected_left_keypoints = left_keypoints[sorted_keypoint_indices[:, 0]]
    selected_right_keypoints = right_keypoints[sorted_keypoint_indices[:, 1]]
    matches = np.array(
        tuple( # get the matches
            selected_left_keypoints[i].pt + selected_right_keypoints[i].pt
            for i in range(len(selected_right_keypoints))
        )
    )
    return matches


def plot_matches(
    matches: np.ndarray,
    left_img: np.ndarray,
    right_img: np.ndarray,
    save_path: str,
):
    total_img = np.concatenate((left_img, right_img), axis=1) # concatenate the images
    match_img = total_img.copy()
    offset = total_img.shape[1] / 2
    _, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.imshow(np.array(match_img).astype("uint8"))  # 　RGB is integer type

    ax.plot(matches[:, 0], matches[:, 1], "xr")
    ax.plot(matches[:, 2] + offset, matches[:, 3], "xr")

    ax.plot( # plot the matches
        [matches[:, 0], matches[:, 2] + offset],
        [matches[:, 1], matches[:, 3]],
        "r",
        linewidth=0.5,
    )
    plt.savefig(os.path.join(save_path, "matches.png"))
    plt.clf()
    plt.close()


def ransac(
    *,
    matches: np.ndarray,
    relative_sample_size: float,
    iters: int,
    distance_threshold: float,
    accuracy_threshold: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # sample size is computed and 10 is the minimum
    sample_size = max(10, int(len(matches) * relative_sample_size))
    print(f"{sample_size=}")
    # define accumulators
    models = []
    errors = []
    accuracies = []
    permuting_indices = []  # used to get the resulting out/in liers
    for _ in tqdm(range(iters)):
        permutation = np.random.permutation(len(matches)) # get a random permutation
        permuted_matches = matches[permutation] # permute the matches
        sample = permuted_matches[:sample_size] # get the sample    
        remaining = permuted_matches[sample_size:] # get the remaining matches
        model = cv2.findHomography(sample[:, :2], sample[:, 2:], cv2.RANSAC)[0] # get the model
        if np.linalg.matrix_rank(model) > 0: # check if the model is invertible
            left = np.concatenate( # get the left and right images
                (remaining[:, :2], np.ones(remaining.shape[0])[:, None]),
                axis=1, # concatenate the images
            )
            # apply the model to all the points
            left_remaining_transformed = np.einsum("ij,kj->ki", model, left)
            left_remaining_transformed = ( # get the transformed points
                left_remaining_transformed[:, :2]
                / left_remaining_transformed[:, -1, None]
            )
            error = ( # get the error
                np.linalg.norm(
                    left_remaining_transformed - remaining[:, 2:], axis=1,
                )
                ** 2
            )
            # compute ratio between inliers and (inliers + outliers)
            accuracy = np.sum(error < distance_threshold) / len(matches)
            accuracies.append(accuracy)
            errors.append( # get the error
                error[error < distance_threshold].mean()
                if (error < distance_threshold).any()
                else float("inf")
            )
            models.append(model)
            permuting_indices.append(permutation)
    accepted_indices = np.flip(np.argsort(accuracies))[
        : max(int(accuracy_threshold * len(accuracies)), 1)
    ] # get the indices of the accepted models
    target_index = np.argmin(np.array(errors)[accepted_indices]) # get the index of the best model
    accuracy = accuracies[target_index]
    error_metric = errors[target_index]
    target_model = np.array(models)[accepted_indices][target_index]
    target_permutation = np.array(permuting_indices)[accepted_indices][
        target_index
    ] # get the permutation of the best model
    target_permuted_matches = matches[target_permutation]
    return ( # get the results
        target_model,
        target_permuted_matches[:sample_size],
        target_permuted_matches[sample_size:],
        accuracy,
        error_metric,
    )


def stitch_imgs_given_model(
    left_img: np.ndarray, right_img: np.ndarray, model: np.ndarray
):
    # convert to float and normalize to avoid noise
    left_img, right_img = (
        cv2.normalize( # normalize the images
            left_img.astype("float"), None, 0.0, 1.0, cv2.NORM_MINMAX
        ),
        cv2.normalize(
            right_img.astype("float"), None, 0.0, 1.0, cv2.NORM_MINMAX
        ),
    )
    # left image
    left_height, left_width, _ = left_img.shape
    left_sizes = np.array([left_width, left_height])
    # compute the corners
    corners = np.array( # get the corners
        [
            [0, 0, 1],
            [left_width, 0, 1],
            [left_width, left_height, 1],
            [0, left_height, 1],
        ]
    )
    # transform them
    corners_transformed = np.einsum("ij,kj->ik", model, corners)
    corners_transformed = corners_transformed[:2] / corners_transformed[2] # get the transformed corners
    mins = np.min(corners_transformed, axis=1) # get the minumum corners
    x_min, y_min = mins
    # compute the translation matrix
    translation_mat = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    new_model = np.einsum("ij,jk->ik", translation_mat, model) # get the new model
    # compute left img warped
    left_warped_size = np.round(np.abs(mins) + left_sizes).astype(int) # get the size of the warped image
    left_warped = cv2.warpPerspective( # get the warped image
        src=left_img, M=new_model, dsize=left_warped_size
    )
    # warp right img
    right_sizes = np.flip(right_img.shape[:2]) # get the sizes of the right image
    right_warped_size = np.round(np.abs(mins) + right_sizes).astype(int) # get the size of the warped image
    right_warped = cv2.warpPerspective( # get the warped image
        src=right_img, M=translation_mat, dsize=right_warped_size # get the warped image
    )
    # Select each pixel so that it comes either from left or right image according
    # to which image has non-black pixel.
    return np.array(
        tuple(
            tuple(
                (
                    left_warped[i, j] # get the pixel from the left image
                    if (
                        np.sum(left_warped[i, j]) > 0
                        and np.sum(right_warped[i, j]) == 0
                    )
                    else (
                        right_warped[i, j] # get the pixel from the right image
                        if (
                            np.sum(right_warped[i, j]) > 0
                            and np.sum(left_warped[i, j]) == 0
                        )
                        else (left_warped[i, j] + right_warped[i, j]) / 2 # get the pixel from the average
                    )
                )
                for j in range(right_warped.shape[1])
            )
            for i in range(right_warped.shape[0])
        )
    )[: right_warped.shape[0], : right_warped.shape[1]]


def stitch(
    *,
    left_img_path: str,
    right_img_path: str,
    do_plot=True,
    save_path: str = ".",
    top_matches: float = 100,
    ransac_iters: int = 100,
    ransac_sample_size: int = 20,
    ransac_distance_threshold: float = 0.1,
    ransac_accuracy_threshold: float = 0.1,
    distance_method: str = "eucledian",
    compute: bool = True,
):
    left_gray_img, left_rgb_img = read_img(left_img_path)
    right_gray_img, right_rgb_img = read_img(right_img_path)

    compute_corner_harris(left_gray_img, save_path)
    # corners are detected better on grayscale
    (left_keypoints, left_descriptors,) = cv2.SIFT_create().detectAndCompute( # get the keypoints and descriptors
        left_gray_img, None
    )
    (right_keypoints, right_descriptors,) = cv2.SIFT_create().detectAndCompute( # get the keypoints and descriptors
        right_gray_img, None
    )

    left_img_with_keypoints = draw_keypoints_on_img(
        left_gray_img, left_rgb_img, left_keypoints
    ) # get the image with keypoints
    right_img_with_keypoints = draw_keypoints_on_img(
        right_gray_img, right_rgb_img, right_keypoints
    ) # get the image with keypoints
    total_kp = np.concatenate(
        (left_img_with_keypoints, right_img_with_keypoints), axis=1
    ) # get the image with keypoints
    plt.imshow(total_kp)
    plt.savefig(os.path.join(save_path, "keypoints.png"))
    plt.clf()
    plt.close()

    matches = get_matches(
        left_descriptors,
        right_descriptors,
        left_keypoints,
        right_keypoints,
        top_matches=top_matches,
        method=distance_method,
    )
    plot_matches(matches, left_rgb_img, right_rgb_img, save_path)

    print(f"Found {len(matches)} matches.")

    model, inliers, outliers, accuracy_metric, error_metric = ransac(
        matches=matches,
        distance_threshold=ransac_distance_threshold,
        accuracy_threshold=ransac_accuracy_threshold,
        relative_sample_size=ransac_sample_size,
        iters=ransac_iters,
    ) # get the model
    # if do_plot:
    #    plot_ransac(model, inliers, outliers)
    if compute:
        stiched_img = stitch_imgs_given_model(
            left_rgb_img, right_rgb_img, model
        ) # get the stiched image
        plt.imshow(stiched_img)
        plt.savefig(os.path.join(save_path, "stitched.png"))
        
    return {"accuracy": accuracy_metric, "error": error_metric}


def print_resuts(args, results):
    # used to export results to markdown
    if tomark_installed:
        print("### Hyperparameters")
        print(
            Tomark.table(
                [
                    {
                        k.replace("_", " "): v
                        for k, v in args.items()
                        if "path" not in k
                    }
                ]
            )
        )
        print("### Results")
        print(Tomark.table([{k: round(v, 5) for k, v in results.items()}]))


def experiment(params):
    # used for testing ranges of values.
    key = "ransac_accuracy_threshold"
    values = np.linspace(0.01, 0.5, 10) # get the values
    results = []
    for val in values: # for each value
        params[key] = val
        result = stitch(**params, compute=False) # get the result
        results.append((val, result["accuracy"], result["error"])) # add the result

    results = np.array(results).T
    plt.clf()
    plt.close()
    keys = ["accuracy", "error"]
    _, axis = plt.subplots(2)
    for i in range(results.shape[0] - 1): # for each key
        axis[i].plot( # plot the results
            results[0], results[i + 1],
        )
        axis[i].set_title(keys[i])
    plt.show()


def main():
    parser = ArgumentParser() # get the arguments
    parser.add_argument(
        "left_img_path",
        type=str,
        help="Path to left image",
        nargs="?",
        default="./sample_images/1l.png",
    )
    parser.add_argument(
        "right_img_path",
        type=str,
        help="Path to right image",
        nargs="?",
        default="./sample_images/1r.png",
    )
    parser.add_argument(
        "--ransac-iters",
        type=int,
        default=10000,
        help="Number of iterations of the Ransac algorithm",
    )
    parser.add_argument(
        "--ransac-sample-size",
        type=float,
        default=0.2,
        help="Relative sample size of Ransac algorithm",
    )
    parser.add_argument(
        "--ransac-distance-threshold",
        type=float,
        default=0.1,
        help="Distance threshold used by Ransac to define inliers and outliers",
    )
    parser.add_argument(
        "--ransac-accuracy-threshold",
        type=float,
        default=0.01,
        help="Relative threshold applied to all models. For example a threshold of 0.1 will select 10 percent best models according to accuracy",
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default="./out_imgs_new2",
        help="Where to save the plots",
    )
    parser.add_argument(
        "--distance-method",
        type=str,
        default="euclidean",
        choices=("euclidean", "correlation"),
        help="Distance method used to find the matches",
    )
    parser.add_argument(
        "--top-matches",
        type=float,
        default=0.04,
        help="Relative threshold applied to the matches. For example a threshold of 0.1 will select 10 percent closest matches.",
    )
    args = parser.parse_args()
    print(json.dumps(args.__dict__, indent=4))
    do_experiment = False
    if not do_experiment:
        results = stitch(**args.__dict__) # get the results
        print_resuts(args.__dict__, results) # print the results
    else:
        experiment(args.__dict__)


if __name__ == "__main__":
    main()
