"""Bounded CPU-only frame preparation for the depth-only video path."""

from collections.abc import Generator, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from queue import Full, Queue
from threading import Event, Thread

import av
import numpy as np
from numpy.typing import NDArray

from nunif.utils.video import to_ndarray

PreparedFrame = tuple[NDArray[np.generic], int | None]


@dataclass
class _Failure:
    error: BaseException


@contextmanager
def prepared_depth_frames(
    frames: Iterable[av.VideoFrame], capacity: int
) -> Generator[Iterator[PreparedFrame], None, None]:
    """Yield ordered RGB arrays/PTS; join the worker before its container closes.

    At most ``capacity`` items are queued plus one producer item in flight.
    The caller owns the input container and all tensor conversion/GPU work.
    """
    if capacity < 1:
        raise ValueError("Depth input queue capacity must be positive")
    pending: Queue[PreparedFrame | _Failure | None] = Queue(maxsize=capacity)
    stop = Event()

    def put(item: PreparedFrame | _Failure | None) -> None:
        while not stop.is_set():
            try:
                pending.put(item, timeout=0.1)
                return
            except Full:
                pass

    def produce() -> None:
        try:
            for frame in frames:
                if stop.is_set():
                    break
                put((to_ndarray(frame), frame.pts))
        except BaseException as error:  # noqa: BLE001
            # Forward worker failures, including cancellation, to the caller.
            put(_Failure(error))
        finally:
            put(None)

    def consume() -> Iterator[PreparedFrame]:
        while True:
            item = pending.get()
            if item is None:
                return
            if isinstance(item, _Failure):
                raise item.error
            yield item

    worker = Thread(target=produce, name="iw3-depth-input")
    worker.start()
    try:
        yield consume()
    finally:
        stop.set()
        worker.join()
