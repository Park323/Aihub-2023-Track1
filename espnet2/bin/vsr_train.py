#!/usr/bin/env python3
from espnet2.tasks.vsr import VSRTask


def get_parser():
    parser = VSRTask.get_parser()
    return parser


def main(cmd=None):
    r"""VSR training.

    Example:

        % python vsr_train.py vsr --print_config --optim adadelta \
                > conf/train_vsr.yaml
        % python vsr_train.py --config conf/train_vsr.yaml
    """
    VSRTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
