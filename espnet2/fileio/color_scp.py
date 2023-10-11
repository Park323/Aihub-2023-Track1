import collections.abc
from pathlib import Path
from typing import Union
from typeguard import check_argument_types

import cv2
import numpy as np

from espnet2.fileio.read_text import read_2columns_text


class Mp4ScpReader(collections.abc.Mapping):
    """Reader class for a scp file of mp4 file.

    Examples:
        key1 /some/path/a.mp4
        key2 /some/path/b.mp4
        key3 /some/path/c.mp4
        key4 /some/path/d.mp4
        ...

        >>> reader = mp4ScpReader('mp4.scp')
        >>> array = reader['key1']

    """

    def __init__(self, fname: Union[Path, str]):
        assert check_argument_types()
        self.fname = Path(fname)
        self.data = read_2columns_text(fname)
        self.fps = None

    def get_path(self, key):
        return self.data[key]

    def get_fps(self):
        return self.fps

    def __getitem__(self, key) -> np.ndarray:
        p = self.data[key]
        
        cap = cv2.VideoCapture(p)
        self.fps = cap.get(cv2.CAP_PROP_FPS)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            frames.append(frame)
        cap.release()
        assert len(frames) > 0, f"Failed to load `{p}`"
        return np.array(frames, dtype=float)

    def __contains__(self, item):
        return item

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def keys(self):
        return self.data.keys()


class Mp4ScpWriter:
    """Writer class for a scp file containing paths to .mp4 files.

    This class saves numpy arrays representing video frames as .mp4 files and
    updates the .scp file with the new file paths.

    Args:
        outdir (Union[Path, str]): Output directory for the .mp4 files.
        scpfile (Union[Path, str]): Output .scp file containing paths to the .mp4 files.
        output_name_format (str, optional): Format for the .mp4 file names. Defaults to "{key}.mp4".
        codec (str, optional): Video codec used for encoding the .mp4 files. Defaults to 'mp4v'.
        fps (int, optional): Frames per second for the .mp4 files. Defaults to 30.

    Examples:
        >>> writer = Mp4ScpWriter('./data/', './data/mp4.scp')
        >>> writer['aa'] = numpy_array
        >>> writer['bb'] = numpy_array
    """

    def __init__(
        self,
        outdir: Union[Path, str],
        scpfile: Union[Path, str],
        output_name_format: str = "{key}.mp4",
        codec: str = 'mp4v',
        fps: int = 30,
    ):
        assert check_argument_types()
        self.dir = Path(outdir)
        self.dir.mkdir(parents=True, exist_ok=True)
        scpfile = Path(scpfile)
        scpfile.parent.mkdir(parents=True, exist_ok=True)
        self.fscp = scpfile.open("w", encoding="utf-8")
        self.output_name_format = output_name_format
        self.codec = codec
        self.fps = fps
        self.data = {}

    def __setitem__(self, key: str, value: np.ndarray):
        assert isinstance(value, np.ndarray), type(value)
        mp4_path = self.dir / self.output_name_format.format(key=key)
        mp4_path.parent.mkdir(parents=True, exist_ok=True)

        fourcc = cv2.VideoWriter_fourcc(*self.codec)
        height, width, _ = value[0].shape
        out = cv2.VideoWriter(str(mp4_path), fourcc, self.fps, (width, height))

        for frame in value:
            out.write(frame)
        out.release()

        self.fscp.write(f"{key} {mp4_path}\n")
        self.data[key] = str(mp4_path)

    def get_path(self, key):
        return self.data[key]

    def get_fps(self):
        return self.fps

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.fscp.close()

    def __getitem__(self, item):
        return self.data[item]

    def __contains__(self, item):
        return item in self.data

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)
