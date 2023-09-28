from argparse import ArgumentParser

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
        "--worm_segments",
        dest="worm_segments",
        type=int,
        default=5,
        help="number of segments in the worm body",
    )
    parser.add_argument(
        "--number_worms",
        dest="number_worms",
        type=int,
        default=1e3,
        help="number of simulated worms",
    )
    args = parser.parse_args()
