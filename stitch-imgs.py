import cv2
from cv2 import KeyPoint
import matplotlib.pyplot as plt
import numpy as np
import random
import ipdb
from argparse import ArgumentParser
from scipy.spatial.distance import cdist
from tqdm import tqdm
import json
import os
from tomark import Tomark
from skimage.feature import corner_harris, corner_subpix, corner_peaks


def compute_corner_harris(image: np.ndarray, save_path: str):

    _, ax = plt.subplots()
    h_corners = corner_harris(image, k=0.02)
    peaks = corner_peaks(h_corners, min_distance=15, threshold_rel=0.01)
    sub_pix = corner_subpix(h_corners, peaks, window_size=10)
    ax.plot(
        peaks[:, 1],
        peaks[:, 0],
        color="cyan",
        marker="o",
        linestyle="None",
        markersize=6,
    )
    ax.plot(sub_pix[:, 1], sub_pix[:, 0], "+r", markersize=15)
    ax.imshow(image, cmap=plt.cm.gray)
    plt.savefig(os.path.join(save_path, "corner_harris.png"))


# coords = corner_peaks(corner_harris(image), min_distance=5, threshold_rel=0.02)
# coords_subpix = corner_subpix(image, coords, window_size=13)

# harris_corner(left_gray)
# harris_corner(right_gray)


def draw_keypoints_on_img(
    gray_img: np.ndarray, rgb_img: np.ndarray, keypoints: list[KeyPoint]
):
    return cv2.drawKeypoints(
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
    max_val = np.max(d1)
    min_val = np.min(d1)
    norm_d1 = (d1 - min_val) / (max_val - min_val)
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
    )
    right_descriptors = (right_descriptors - min_desc_value) / (
        max_desc_value - min_desc_value
    )
    # computes the distance between the normalized descriptors using cdist
    distances = cdist(left_descriptors, right_descriptors, method)
    sorting_indices = np.stack(
        (np.arange(len(distances)), np.argmin(distances, axis=1)), axis=1
    )
    row_sorting = np.argsort(
        distances[sorting_indices[:, 0], sorting_indices[:, 1]]
    )
    max_matches = int(len(sorting_indices) * top_matches)
    sorted_keypoint_indices = sorting_indices[row_sorting][:max_matches]
    selected_left_keypoints = left_keypoints[sorted_keypoint_indices[:, 0]]
    selected_right_keypoints = right_keypoints[sorted_keypoint_indices[:, 1]]
    matches = np.array(
        tuple(
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
    total_img = np.concatenate((left_img, right_img), axis=1)
    match_img = total_img.copy()
    offset = total_img.shape[1] / 2
    fig, ax = plt.subplots()
    ax.set_aspect("equal")
    ax.imshow(np.array(match_img).astype("uint8"))  # 　RGB is integer type

    ax.plot(matches[:, 0], matches[:, 1], "xr")
    ax.plot(matches[:, 2] + offset, matches[:, 3], "xr")

    ax.plot(
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
    accepted_threshold: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # sample size is computed and 10 is the minimum
    sample_size = max(10, int(len(matches) * relative_sample_size))
    print(f"{sample_size=}")
    # define accumulators
    models = []
    errors = []
    respecting_vals = []
    permuting_indices = []  # used to get the resulting out/in liers
    for _ in tqdm(range(iters)):
        permutation = np.random.permutation(len(matches))
        permuted_matches = matches[permutation]
        sample = permuted_matches[:sample_size]
        remaining = permuted_matches[sample_size:]
        model = cv2.findHomography(sample[:, :2], sample[:, 2:], cv2.RANSAC)[0]
        if np.linalg.matrix_rank(model) > 0:
            left = np.concatenate(
                (remaining[:, :2], np.ones(remaining.shape[0])[:, None]),
                axis=1,
            )
            # apply the model to all the points
            left_remaining_transformed = np.einsum("ij,kj->ki", model, left)
            left_remaining_transformed = (
                left_remaining_transformed[:, :2]
                / left_remaining_transformed[:, -1, None]
            )
            error = (
                np.linalg.norm(
                    left_remaining_transformed - remaining[:, 2:], axis=1,
                )
                ** 2
            )
            # compute ratio between inliers and outliers
            respecting = np.sum(error < distance_threshold) / len(error)
            respecting_vals.append(respecting)
            errors.append(
                error[error < distance_threshold].mean()
                if (error < distance_threshold).any()
                else float("inf")
            )
            models.append(model)
            permuting_indices.append(permutation)
    accepted_indices = np.flip(np.argsort(respecting_vals))[
        : max(int(accepted_threshold * len(respecting_vals)), 1)
    ]
    target_index = np.argmin(np.array(errors)[accepted_indices])
    respecting_metric = respecting_vals[target_index]
    error_metric = errors[target_index]
    target_model = np.array(models)[accepted_indices][target_index]
    target_permutation = np.array(permuting_indices)[accepted_indices][
        target_index
    ]
    target_permuted_matches = matches[target_permutation]
    return (
        target_model,
        target_permuted_matches[:sample_size],
        target_permuted_matches[sample_size:],
        respecting_metric,
        error_metric,
    )


def stitch_imgs_given_model(
    left_img: np.ndarray, right_img: np.ndarray, model: np.ndarray
):
    # convert to float and normalize to avoid noise
    left_img, right_img = (
        cv2.normalize(
            left_img.astype("float"), None, 0.0, 1.0, cv2.NORM_MINMAX
        ),
        cv2.normalize(
            right_img.astype("float"), None, 0.0, 1.0, cv2.NORM_MINMAX
        ),
    )
    # left image
    left_height, left_width, _ = left_img.shape
    left_sizes = np.array([left_width, left_height])
    corners = np.array(
        [
            [0, 0, 1],
            [left_width, 0, 1],
            [left_width, left_height, 1],
            [0, left_height, 1],
        ]
    )
    corners_transformed = np.einsum("ij,kj->ik", model, corners)
    corners_transformed = corners_transformed[:2] / corners_transformed[2]
    mins = np.min(corners_transformed, axis=1)
    x_min, y_min = mins
    translation_mat = np.array([[1, 0, -x_min], [0, 1, -y_min], [0, 0, 1]])
    new_model = np.einsum("ij,jk->ik", translation_mat, model)
    left_warped_size = np.round(np.abs(mins) + left_sizes).astype(int)
    left_warped = cv2.warpPerspective(
        src=left_img, M=new_model, dsize=left_warped_size
    )

    # right image
    right_sizes = np.flip(right_img.shape[:2])
    right_warped_size = np.round(np.abs(mins) + right_sizes).astype(int)
    right_warped = cv2.warpPerspective(
        src=right_img, M=translation_mat, dsize=right_warped_size
    )

    return np.array(
        tuple(
            tuple(
                (
                    left_warped[i, j]
                    if (
                        np.sum(left_warped[i, j]) > 0
                        and np.sum(right_warped[i, j]) == 0
                    )
                    else (
                        right_warped[i, j]
                        if (
                            np.sum(right_warped[i, j]) > 0
                            and np.sum(left_warped[i, j]) == 0
                        )
                        else (left_warped[i, j] + right_warped[i, j]) / 2
                    )
                )
                for j in range(right_warped.shape[1])
            )
            for i in range(right_warped.shape[0])
        )
    )[: right_warped.shape[0], : right_warped.shape[1]]


def plot_ransac(model: np.ndarray, inliers, outliers):
    plt.clf()
    plt.close()
    # ax = plt.axes(projection="3d")
    inliers = np.transpose(inliers, (1, 0))
    inliers = np.concatenate((inliers[:2], inliers[2:]), axis=1)
    outliers = np.transpose(outliers, (1, 0))
    outliers = np.concatenate((outliers[:2], outliers[2:]), axis=1)
    plt.plot(*inliers, "o")
    plt.plot(*outliers, "o")
    mins = np.min(np.concatenate((inliers, outliers), axis=1), axis=1)
    maxs = np.max(np.concatenate((inliers, outliers), axis=1), axis=1)
    proto_mesh = np.meshgrid(
        np.linspace(mins[0], 2 * maxs[1], 100),
        np.linspace(mins[1], 2 * maxs[1], 100),
    )[0]
    mesh = np.stack(
        (proto_mesh[0], proto_mesh[1], np.ones(proto_mesh.shape[1]))
    ).T
    transformed_mesh = np.einsum("ij,kj->ik", model, mesh)
    transformed_mesh = transformed_mesh[:2] / transformed_mesh[None, -1]
    min_x_filer = transformed_mesh[0] >= mins[0]
    max_x_filter = transformed_mesh[0] >= maxs[0]
    min_y_filter = transformed_mesh[1] >= mins[1]
    max_y_filter = transformed_mesh[1] >= maxs[1]
    total_filter = min_x_filer & max_x_filter & min_y_filter & max_y_filter
    transformed_mesh = transformed_mesh[:, total_filter]
    plt.plot(*transformed_mesh, "--")
    # ax.scatter3D(*inliers, label="inliers")
    # ax.scatter3D(*outliers, label="outliers")
    # mins = np.min(np.min(inliers), np.min(outliers))
    # maxs = np.max(np.max(inliers), np.max(outliers))
    plt.legend()
    plt.show()


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
    ransac_accepted_threshold: float = 0.1,
    distance_method: str = "eucledian",
):
    left_gray_img, left_rgb_img = read_img(left_img_path)
    right_gray_img, right_rgb_img = read_img(right_img_path)

    compute_corner_harris(left_gray_img, save_path)
    # corners are detected better on grayscale
    (left_keypoints, left_descriptors,) = cv2.SIFT_create().detectAndCompute(
        left_gray_img, None
    )
    (right_keypoints, right_descriptors,) = cv2.SIFT_create().detectAndCompute(
        right_gray_img, None
    )

    left_img_with_keypoints = draw_keypoints_on_img(
        left_gray_img, left_rgb_img, left_keypoints
    )
    right_img_with_keypoints = draw_keypoints_on_img(
        right_gray_img, right_rgb_img, right_keypoints
    )
    total_kp = np.concatenate(
        (left_img_with_keypoints, right_img_with_keypoints), axis=1
    )
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
        accepted_threshold=ransac_accepted_threshold,
        relative_sample_size=ransac_sample_size,
        iters=ransac_iters,
    )
    # if do_plot:
    #    plot_ransac(model, inliers, outliers)
    stiched_img = stitch_imgs_given_model(left_rgb_img, right_rgb_img, model)
    plt.imshow(stiched_img)
    plt.savefig(os.path.join(save_path, "stitched.png"))
    return {"accuracy": accuracy_metric, "error": error_metric}


def print_resuts(args, results):
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


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "left_img_path",
        type=str,
        help="Path to left image",
        nargs="?",
        default="./sample_images/2l.jpg",
    )
    parser.add_argument(
        "right_img_path",
        type=str,
        help="Path to right image",
        nargs="?",
        default="./sample_images/2r.jpg",
    )
    parser.add_argument("--ransac-iters", type=int, default=10000)
    parser.add_argument("--ransac-sample-size", type=float, default=0.2)
    parser.add_argument("--ransac-distance-threshold", type=float, default=0.1)
    parser.add_argument(
        "--ransac-accepted-threshold", type=float, default=0.01
    )
    parser.add_argument("--save-path", type=str, default="./out_imgs")
    parser.add_argument("--distance-method", type=str, default="euclidean")
    # parser.add_argument('--experiment', type=bool, default=False)
    parser.add_argument("--top-matches", type=float, default=0.04)
    args = parser.parse_args()
    print(json.dumps(args.__dict__, indent=4))
    # if args.experiment:
    #    pass
    results = stitch(**args.__dict__)
    print_resuts(args.__dict__, results)


if __name__ == "__main__":
    main()
