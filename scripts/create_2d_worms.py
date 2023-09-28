from argparse import ArgumentParser

import numpy as np
from tqdm import tqdm


def create_worms(**args):
    worm_dimension = args["worm_dimension"]
    number_segments = args["number_segments"]
    number_worms = args["number_worms"]
    output_dir_path = args["output_dir_path"]
    s_w = args["straightened_width"]
    s_l = args["straightened_length"]
    print(
        f"worm_dimension {worm_dimension}, \
        number_segments {number_segments}, \
        number_worms {number_worms}, \
        output path {output_dir_path}"
    )

    scale_mu = 1.0
    theta_mu = 0.0
    t_mu = s_l / number_segments
    scale_sigma = scale_mu / 5.0
    theta_sigma = np.pi / 8
    t_sigma = t_mu / 5.0

    scale_list = [None] * number_segments
    theta_list = [None] * number_segments
    t_list = [None] * number_segments

    for i in tqdm(range(number_worms)):
        x = np.array([[0, 0 + s_w], [0, 0]])  # 2D
        for j in range(number_segments):
            scale = np.random.normal(scale_mu, scale_sigma, worm_dimension)
            theta = np.random.normal(theta_mu, theta_sigma)
            t = np.random.normal(t_mu, t_sigma, worm_dimension)

            print(f"Scale is {scale}, theta is {theta}, t is {t}")

            scale_list[j] = scale
            theta_list[j] = theta
            t_list[j] = t

            scale_matrix = np.array([[scale[0], 0], [0, scale[1]]])
            theta_matrix = np.array(
                [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
            )
            np.array([[t[0]], [t[1]]])
            sR = np.matmul(scale_matrix, theta_matrix)
            sRx = np.matmul(sR, x)
            x = sRx + t
            print(x)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--worm_dimension",
        dest="worm_dimension",
        type=int,
        default=2,
        help="worm lives in n-D space",
    )
    parser.add_argument(
        "--number_segments",
        dest="number_segments",
        type=int,
        default=5,
        help="number of segments in the worm body",
    )
    parser.add_argument(
        "--number_worms",
        dest="number_worms",
        type=int,
        default=1,
        help="number of simulated worms",
    )
    parser.add_argument(
        "--output_dir_path",
        dest="output_dir_path",
        type=str,
        default="/Users/lalitm/Desktop/simulated_data",
        help="where all the simulated worms are saved",
    )
    parser.add_argument(
        "--straightened_width",
        dest="straightened_width",
        type=float,
        default=1.0,
        help="straightened width",
    )

    parser.add_argument(
        "--straightened_length",
        dest="straightened_length",
        type=float,
        default=10.0,
        help="straightened length",
    )

    args = parser.parse_args()
    create_worms(**vars(args))
