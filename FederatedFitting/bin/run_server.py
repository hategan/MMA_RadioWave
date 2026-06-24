import argparse
import logging

from mma_fedfit.server.server import Server


logging.basicConfig(level=logging.WARNING)


def parse_args() -> argparse.Namespace:
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--config",
        type=str,
        default="examples/configs/server.yaml",
        help="Path to the configuration file."
    )

    return argparser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    server = Server(args.config)
    server.run()

