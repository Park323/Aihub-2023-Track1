import logging
from random import shuffle
from typing import Dict, Iterator, List, Optional, Tuple, Union

from typeguard import check_argument_types


from espnet2.fileio.read_text import load_num_sequence_text
from espnet2.samplers.abs_sampler import AbsSampler


class LengthBatchSampler(AbsSampler):
    def __init__(
        self,
        batch_bins: int,
        shape_files: Union[Tuple[str, ...], List[str]],
        max_utt_len: int = None,
        min_batch_size: int = 1,
        sort_in_batch: str = "descending",
        sort_batch: Optional[str] = "ascending",
        drop_last: bool = False,
        padding: bool = True,
    ):
        assert check_argument_types()
        assert batch_bins > 0
        if sort_batch != "ascending" and sort_batch != "descending" and sort_batch is not None:
            raise ValueError(
                f"sort_batch must be ascending or descending: {sort_batch}"
            )
        if sort_in_batch != "descending" and sort_in_batch != "ascending" and sort_in_batch is not None:
            raise ValueError(
                f"sort_in_batch must be ascending or descending: {sort_in_batch}"
            )

        self.batch_bins = batch_bins
        self.shape_files = shape_files
        self.max_utt_len = max_utt_len
        self.min_batch_size = min_batch_size
        self.sort_in_batch = sort_in_batch
        self.sort_batch = sort_batch
        self.drop_last = drop_last
        self.padding = padding

        # utt2shape: (Length, ...)
        #    uttA 100,...
        #    uttB 201,...
        utt2shapes = [
            load_num_sequence_text(s, loader_type="csv_int") for s in shape_files
        ]

        if max_utt_len is not None:
            # utt2shapes : [{key:shape}, ...]
            sub_utt2shapes = [{} for _ in utt2shapes]
            for k, v in utt2shapes[0].items():
                if v[0] < max_utt_len:
                    for i, utt2shape in enumerate(utt2shapes):
                        sub_utt2shapes[i][k] = utt2shape[k]
            
            logging.info(f"Sampled N-item: {len(sub_utt2shapes[0])}/{len(utt2shapes[0])}")

            utt2shapes = sub_utt2shapes
            
        first_utt2shape = utt2shapes[0]
        for s, d in zip(shape_files, utt2shapes):
            if set(d) != set(first_utt2shape):
                raise RuntimeError(
                    f"keys are mismatched between {s} != {shape_files[0]}"
                )

        # Sort samples in ascending order
        # (shape order should be like (Length, Dim))
        if sort_batch is not None:
            keys = sorted(first_utt2shape, key=lambda k: first_utt2shape[k][0])
        else:
            keys = list(first_utt2shape.keys())
            shuffle(keys)
        if len(keys) == 0:
            raise RuntimeError(f"0 lines found: {shape_files[0]}")

        # Decide batch-sizes
        batch_sizes = []
        current_batch_keys = []
        for key in keys:
            current_batch_keys.append(key)
            # shape: (Length, dim1, dim2, ...)
            if padding:
                # bins = bs x max_length
                bins = sum(len(current_batch_keys) * sh[key][0] for sh in utt2shapes)
            else:
                # bins = sum of lengths
                bins = sum(d[k][0] for k in current_batch_keys for d in utt2shapes)

            if bins > batch_bins and len(current_batch_keys) >= min_batch_size:
                batch_sizes.append(len(current_batch_keys) - 1)
                current_batch_keys = [key]
        else:
            if len(current_batch_keys) != 0 and (
                not self.drop_last or len(batch_sizes) == 0
            ):
                batch_sizes.append(len(current_batch_keys))

        if len(batch_sizes) == 0:
            # Maybe we can't reach here
            raise RuntimeError("0 batches")

        # If the last batch-size is smaller than minimum batch_size,
        # the samples are redistributed to the other mini-batches
        if len(batch_sizes) > 1 and batch_sizes[-1] < min_batch_size:
            for i in range(batch_sizes.pop(-1)):
                batch_sizes[-(i % len(batch_sizes)) - 1] += 1

        if not self.drop_last:
            # Bug check
            assert sum(batch_sizes) == len(keys), f"{sum(batch_sizes)} != {len(keys)}"

        # Set mini-batch
        self.batch_list = []
        iter_bs = iter(batch_sizes)
        bs = next(iter_bs)
        minibatch_keys = []
        for key in keys:
            minibatch_keys.append(key)
            if len(minibatch_keys) == bs:
                if sort_in_batch == "descending":
                    minibatch_keys.reverse()
                elif sort_in_batch == "ascending":
                    # Key are already sorted in ascending
                    pass
                else:
                    raise ValueError(
                        "sort_in_batch must be ascending"
                        f" or descending: {sort_in_batch}"
                    )
                self.batch_list.append(tuple(minibatch_keys))
                minibatch_keys = []
                try:
                    bs = next(iter_bs)
                except StopIteration:
                    break

        if sort_batch is None or sort_batch == "ascending":
            pass
        elif sort_batch == "descending":
            self.batch_list.reverse()
        else:
            raise ValueError(
                f"sort_batch must be ascending or descending: {sort_batch}"
            )
        
        logging.info(f"Subsampled N-batch: {len(self)}")

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"N-batch={len(self)}, "
            f"batch_bins={self.batch_bins}, "
            f"sort_in_batch={self.sort_in_batch}, "
            f"sort_batch={self.sort_batch})"
        )

    def __len__(self):
        return len(self.batch_list)

    def __iter__(self) -> Iterator[Tuple[str, ...]]:
        return iter(self.batch_list)
    
    def distribute_batches(self, rank:int, world_size:int):
        self.batch_list = self.batch_list[rank::world_size]
        logging.info(f"The number of batches per device : {len(self.batch_list)}")
    
    def generate(self, epoch, *args):
        if self.sort_batch is None:
            raise NotImplementedError("Implementation Is Not Completed")
            logging.info(f"Shuffle the batches for {epoch}th epoch.")
            self.__init__(
                batch_bins=self.batch_bins,
                shape_files=self.shape_files,
                max_utt_len=self.max_utt_len,
                min_batch_size=self.min_batch_size,
                sort_in_batch=self.sort_in_batch,
                sort_batch=self.sort_batch,
                drop_last=self.drop_last,
                padding=self.padding
            )
        return super().generate(*args)


class LengthBatchCurriculum(AbsSampler):
    """
        Example
        >>> curriculum_epochs = [1, 51, 101]
        >>> curriculum_max_len = [16000, 90000, 160000]
    """
    def __init__(
        self,
        batch_bins: int,
        shape_files: Union[Tuple[str, ...], List[str]],
        curriculum_epochs: Optional[List[int]],
        curriculum_max_len: Optional[List[int]],
        min_batch_size: int = 1,
        sort_in_batch: str = "descending",
        sort_batch: str = "ascending",
        drop_last: bool = False,
        padding: bool = True,
    ):
        logging.warning(f"{self.__class__} is deprecated. Use `max_utt_len` argument in LengthBatchSampler for curriculum learning")
        assert check_argument_types()
        if curriculum_epochs is None: curriculum_epochs = []
        if curriculum_max_len is None: curriculum_max_len = []
        assert len(curriculum_epochs) == len(curriculum_max_len)
        assert is_increasing(curriculum_epochs), f"curriculum epochs should be increasing: {curriculum_epochs}"
        assert batch_bins > 0
        if sort_batch != "ascending" and sort_batch != "descending":
            raise ValueError(
                f"sort_batch must be ascending or descending: {sort_batch}"
            )
        if sort_in_batch != "descending" and sort_in_batch != "ascending":
            raise ValueError(
                f"sort_in_batch must be ascending or descending: {sort_in_batch}"
            )

        self.batch_bins = batch_bins
        self.shape_files = shape_files
        self.sort_in_batch = sort_in_batch
        self.sort_batch = sort_batch
        self.drop_last = drop_last

        self.min_batch_size = min_batch_size
        self.padding = padding

        self.curriculum_epochs = curriculum_epochs if curriculum_epochs is not None else []
        self.curriculum_max_len = curriculum_max_len

        self.max_len = None
        
        # utt2shape: (Length, ...)
        #    uttA 100,...
        #    uttB 201,...
        self.utt2shapes = [
            load_num_sequence_text(s, loader_type="csv_int") for s in shape_files
        ]

        self.build_batch(self.utt2shapes)
        
    def build_batch(self, utt2shapes: List[Dict[str, int]]):

        first_utt2shape = utt2shapes[0]
        for s, d in zip(self.shape_files, utt2shapes):
            if set(d) != set(first_utt2shape):
                raise RuntimeError(
                    f"keys are mismatched between {s} != {self.shape_files[0]}"
                )

        # Sort samples in ascending order
        # (shape order should be like (Length, Dim))
        keys = sorted(first_utt2shape, key=lambda k: first_utt2shape[k][0])
        if len(keys) == 0:
            raise RuntimeError(f"0 lines found: {self.shape_files[0]}")

        # Decide batch-sizes
        batch_sizes = []
        current_batch_keys = []
        for key in keys:
            current_batch_keys.append(key)
            # shape: (Length, dim1, dim2, ...)
            if self.padding:
                # bins = bs x max_length
                bins = sum(len(current_batch_keys) * sh[key][0] for sh in utt2shapes)
            else:
                # bins = sum of lengths
                bins = sum(d[k][0] for k in current_batch_keys for d in utt2shapes)
            
            if bins > self.batch_bins and len(current_batch_keys) >= self.min_batch_size:
                batch_sizes.append(len(current_batch_keys))
                current_batch_keys = []
        else:
            if len(current_batch_keys) != 0 and (
                not self.drop_last or len(batch_sizes) == 0
            ):
                batch_sizes.append(len(current_batch_keys))

        if len(batch_sizes) == 0:
            # Maybe we can't reach here
            raise RuntimeError("0 batches")

        # If the last batch-size is smaller than minimum batch_size,
        # the samples are redistributed to the other mini-batches
        if len(batch_sizes) > 1 and batch_sizes[-1] < self.min_batch_size:
            for i in range(batch_sizes.pop(-1)):
                batch_sizes[-(i % len(batch_sizes)) - 1] += 1

        if not self.drop_last:
            # Bug check
            assert sum(batch_sizes) == len(keys), f"{sum(batch_sizes)} != {len(keys)}"

        # Set mini-batch
        self.batch_list = []
        iter_bs = iter(batch_sizes)
        bs = next(iter_bs)
        minibatch_keys = []
        for key in keys:
            minibatch_keys.append(key)
            if len(minibatch_keys) == bs:
                if self.sort_in_batch == "descending":
                    minibatch_keys.reverse()
                elif self.sort_in_batch == "ascending":
                    # Key are already sorted in ascending
                    pass
                else:
                    raise ValueError(
                        "sort_in_batch must be ascending"
                        f" or descending: {self.sort_in_batch}"
                    )

                self.batch_list.append(tuple(minibatch_keys))
                minibatch_keys = []
                try:
                    bs = next(iter_bs)
                except StopIteration:
                    break

        if self.sort_batch == "ascending":
            pass
        elif self.sort_batch == "descending":
            self.batch_list.reverse()
        else:
            raise ValueError(
                f"sort_batch must be ascending or descending: {self.sort_batch}"
            )

    def build_subbatch(self):
        # self.utt2shapes : [{key:shape}, ...]
        sub_utt2shapes = [{} for _ in self.utt2shapes]
        for k, v in self.utt2shapes[0].items():
            if v[0] < self.max_len:
                for i, utt2shape in enumerate(self.utt2shapes):
                    sub_utt2shapes[i][k] = utt2shape[k]
        
        logging.info(f"Subsampled N-item: {len(sub_utt2shapes[0])}/{len(self.utt2shapes[0])}")
        
        ex_batchlen = len(self)
        self.build_batch(sub_utt2shapes)
        logging.info(f"Subsampled N-batch: {ex_batchlen} -> {len(self)}")

    def distribute_batches(self, rank:int, world_size:int):
        sub_utt2shapes = []
        for utt2shape in self.utt2shapes:
            distributed_shapes = {}
            for i, (k, v) in enumerate(utt2shape.items()):
                if i % world_size == rank:
                    distributed_shapes[k] = v
            sub_utt2shapes.append(distributed_shapes)
        self.utt2shapes = sub_utt2shapes

    def set_max_len(self, epoch):
        curriculum_epochs = self.curriculum_epochs.copy()
        for c_epoch in curriculum_epochs:
            if epoch >= c_epoch:
                self.curriculum_epochs.pop(0)
                self.max_len = self.curriculum_max_len.pop(0)
                logging.info(f"Build a new sampler for {c_epoch}th epoch.")
                logging.info(f"Limit the max length by {self.max_len}.")
                self.build_subbatch()
            else:
                break

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"N-batch={len(self)}, "
            f"batch_bins={self.batch_bins}, "
            f"sort_in_batch={self.sort_in_batch}, "
            f"sort_batch={self.sort_batch})"
        )

    def __len__(self):
        return len(self.batch_list)

    def __iter__(self) -> Iterator[Tuple[str, ...]]:
        return iter(self.batch_list)

    def generate(self, epoch, *args):
        logging.info(f"Generate a batch list for {epoch}th epoch.")
        self.set_max_len(epoch)
        return super().generate(*args)


def is_increasing(values:List[int]):
    prev = None
    for value in values:
        if prev is None:
            pass
        elif value <= prev:
            return False
        prev = value
    return True