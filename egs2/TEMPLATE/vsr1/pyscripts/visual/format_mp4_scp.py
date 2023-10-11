#!/usr/bin/env python3
import argparse
import logging
from io import BytesIO
from pathlib import Path
from typing import Optional, Tuple

import humanfriendly
import kaldiio
import numpy as np
import resampy
from tqdm import tqdm
from typeguard import check_argument_types

from espnet2.fileio.read_text import read_2columns_text
from espnet2.fileio.mp4_scp import Mp4ScpWriter
from espnet2.fileio.vad_scp import VADScpReader
from espnet2.utils.types import str2bool
from espnet.utils.cli_utils import get_commandline_args


def humanfriendly_or_none(value: str):
    if value in ("none", "None", "NONE"):
        return None
    return humanfriendly.parse_size(value)


def str2int_tuple(integers: str) -> Optional[Tuple[int, ...]]:
    """

    >>> str2int_tuple('3,4,5')
    (3, 4, 5)

    """
    assert check_argument_types()
    if integers.strip() in ("none", "None", "NONE", "null", "Null", "NULL"):
        return None
    return tuple(map(int, integers.strip().split(",")))


def vad_trim(vad_reader: VADScpReader, uttid: str, mp4: np.array, fps: int) -> np.array:
    # Conduct trim wtih vad information

    assert check_argument_types()
    assert uttid in vad_reader, uttid

    vad_info = vad_reader[uttid]
    total_length = sum(int((time[1] - time[0]) * fps) for time in vad_info)
    new_mp4 = np.zeros((total_length,), dtype=mp4.dtype)
    start_frame = 0
    for time in vad_info:
        # Note: we regard vad as [xxx, yyy)
        duration = int((time[1] - time[0]) * fps)
        orig_start_frame = int(time[0] * fps)
        orig_end_frame = orig_start_frame + duration

        end_frame = start_frame + duration
        new_mp4[start_frame:end_frame] = mp4[orig_start_frame:orig_end_frame]

        start_frame = end_frame

    return new_mp4


class SegmentsExtractor:
    """Emulating kaldi extract-segments.cc

    Args:
        segments (str): The file format is
            "<segment-id> <recording-id> <start-time> <end-time>\n"
            "e.g. call-861225-A-0050-0065 call-861225-A 5.0 6.5\n"
    """

    def __init__(self, fname: str, segments: str = None, multi_columns: bool = False):
        assert check_argument_types()
        self.mp4_scp = fname
        self.multi_columns = multi_columns
        self.mp4_dict = {}
        with open(self.mp4_scp, "r") as f:
            for line in f:
                recodeid, mp4path = line.strip().split(None, 1)
                if recodeid in self.mp4_dict:
                    raise RuntimeError(f"{recodeid} is duplicated")
                self.mp4_dict[recodeid] = mp4path

        self.segments = segments
        self.segments_dict = {}
        with open(self.segments, "r") as f:
            for line in f:
                sps = line.rstrip().split(None)
                if len(sps) != 4:
                    raise RuntimeError("Format is invalid: {}".format(line))
                uttid, recodeid, st, et = sps
                self.segments_dict[uttid] = (recodeid, float(st), float(et))

                if recodeid not in self.mp4_dict:
                    raise RuntimeError(
                        'Not found "{}" in {}'.format(recodeid, self.mp4_scp)
                    )

    def generator(self):
        recodeid_counter = {}
        for utt, (recodeid, st, et) in self.segments_dict.items():
            recodeid_counter[recodeid] = recodeid_counter.get(recodeid, 0) + 1

        cached = {}
        for utt, (recodeid, st, et) in self.segments_dict.items():
            if recodeid not in cached:
                mp4path = self.mp4_dict[recodeid]

                if mp4path.endswith("|"):
                    if self.multi_columns:
                        raise RuntimeError(
                            "Not supporting multi_columns mp4.scp for inputs by pipe"
                        )
                    # Streaming input e.g. cat a.mp4 |
                    with kaldiio.open_like_kaldi(mp4path, "rb") as f:
                        with BytesIO(f.read()) as g:
                            retval = soundfile.read(g)
                else:
                    if self.multi_columns:
                        retval = soundfile_read(
                            mp4s=mp4path.split(),
                            dtype=None,
                            always_2d=False,
                            concat_axis=1,
                        )
                    else:
                        retval = soundfile.read(mp4path)

                cached[recodeid] = retval

            # Keep array until the last query
            recodeid_counter[recodeid] -= 1
            if recodeid_counter[recodeid] == 0:
                cached.pop(recodeid)

            yield utt, self._return(retval, st, et), None, None

    def _return(self, array, st, et):
        if isinstance(array, (tuple, list)):
            array, rate = array

        # Convert starting time of the segment to corresponding sample number.
        # If end time is -1 then use the whole file starting from start time.
        if et != -1:
            return array[int(st * rate) : int(et * rate)], rate
        else:
            return array[int(st * rate) :], rate


def main():
    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=logfmt)
    logging.info(get_commandline_args())

    parser = argparse.ArgumentParser(
        description='Create mp4s list from "mp4.scp"',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("scp")
    parser.add_argument("outdir")
    parser.add_argument(
        "--name",
        default="mp4",
        help='Specify the prefix word of output file name such as "mp4.scp"',
    )
    parser.add_argument("--segments", default=None)
    parser.add_argument(
        "--fps",
        type=humanfriendly_or_none,
        default=None,
        help="If the sampling rate specified, Change the sampling rate.",
    )
    parser.add_argument("--visual-format", default="mp4")
    parser.add_argument("--vad_based_trim", type=str, default=None)
    group = parser.add_mutually_exclusive_group()
    parser.add_argument(
        "--multi-columns-input",
        type=str2bool,
        default=False,
        help=(
            "Enable multi columns mode for input wav.scp. "
            "e.g. 'ID a.wav b.wav c.wav' is interpreted as 3ch audio data"
        ),
    )
    parser.add_argument(
        "--multi-columns-output",
        type=str2bool,
        default=False,
        help=(
            "Enable multi columns mode for output wav.scp. "
            "e.g. If input audio data has 2ch, "
            "each line in wav.scp has the the format like "
            "'ID ID-CH0.wav ID-CH1.wav'"
        ),
    )
    args = parser.parse_args()

    out_num_samples = Path(args.outdir) / "utt2num_samples"

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    out_mp4scp = Path(args.outdir) / f"{args.name}.scp"

    writer = Mp4ScpWriter(
        args.outdir,
        out_mp4scp,
        output_name_format="{key}/."+args.visual_format,
        fps=args.fps
    )
    fscp_out = None

    if args.vad_based_trim is not None:
        raise NotImplementedError
        vad_reader = VADScpReader(args.vad_based_trim)
    
    if args.segments is not None:
        raise NotImplementedError
        extractor = SegmentsExtractor(
            args.scp, segments=args.segments, multi_columns=args.multi_columns_input
        )
        generator = extractor.generator

    else:

        def generator():
            with Path(args.scp).open("r") as fscp:
                for line in tqdm(fscp):
                    uttid, mp4path = line.strip().split(None, 1)

                    # B.a. Without segments and using pipe inputs
                    if mp4path.endswith("|"):
                        # Streaming input e.g. cat a.mp4 |
                        raise NotImplementedError
                        subtypes = None

                    # B.b Without segments and not using pipe
                    else:
                        if args.multi_columns_input:
                            raise NotImplementedError
                        else:
                            import cv2
                            cap = cv2.VideoCapture(mp4path)
                            rate = cap.get(cv2.CAP_PROP_FPS)
                            frames = []
                            while cap.isOpened():
                                ret, frame = cap.read()
                                if not ret: break
                                grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                                frames.append(grayframe)
                            cap.release()
                            video = np.array(frames, dtype=float)
                    yield uttid, (video, rate), mp4path

    with out_num_samples.open("w") as fnum_samples:
        for uttid, (video, rate), mp4path in tqdm(generator()):
            save_asis = True
            if args.fps is not None and args.fps != rate:
                # FIXME(kamo): To use sox?
                raise NotImplementedError
                video = resampy.resample(video, rate, args.fps, axis=0)
                rate = args.fps
                save_asis = False

            if args.vad_based_trim is not None:
                raise NotImplementedError
                mp4e = vad_trim(vad_reader, uttid, mp4e, rate)
                save_asis = False

            if args.segments is not None:
                save_asis = False

            if args.multi_columns_input:
                if args.multi_columns_output:
                    if mp4path is not None:
                        for _mp4path in mp4path.split():
                            if Path(_mp4path).suffix != "." + args.visual_format:
                                save_asis = False
                                break

                        if video.ndim == 1:
                            _num_ch = 1
                        else:
                            _num_ch = video.shape[1]
                        if len(mp4path.split()) != _num_ch:
                            save_asis = False
                else:
                    if mp4path is not None and len(mp4path.split()) > 1:
                        save_asis = False

            elif args.multi_columns_output:
                if video.ndim == 2 and video.shape[1] > 1:
                    save_asis = False

            if mp4path is not None and mp4path.endswith("|"):
                save_asis = False
            if mp4path is not None and Path(mp4path).suffix != "." + args.visual_format:
                save_asis = False

            if save_asis:
                writer.fscp.write(f"{uttid} {mp4path}\n")

            else:
                writer[uttid] = rate, video
            fnum_samples.write(f"{uttid} {len(video)}\n")


if __name__ == "__main__":
    main()
