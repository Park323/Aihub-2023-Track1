#!/usr/bin/env python3
from espnet2.tasks.avsr import AVSRTask


def get_parser():
    parser = AVSRTask.get_parser()
    return parser


def main(cmd=None):
    r"""AVSR training.

    Example:

        % python avsr_train.py avsr --print_config --optim adadelta \
                > conf/train_avsr.yaml
        % python avsr_train.py --config conf/train_avsr.yaml
    """
    AVSRTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
