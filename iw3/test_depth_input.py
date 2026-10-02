import tempfile
import threading
import unittest
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import av
import numpy as np
import torch

from iw3 import depth_input, utils
from nunif.utils.video import to_ndarray


def frame(index):
    value = av.VideoFrame.from_ndarray(np.full((8, 8, 3), 32 + index, dtype=np.uint8), format="rgb24")
    value.pts = 10 + index
    return value


class SceneBoundaryClockTest(unittest.TestCase):
    def test_millisecond_timestamps_and_concatenated_rounding(self):
        points = [5139, 5153, 5167, 10306, 10320, 10334]
        boundaries = {371, 743}
        actual = utils._depth_only_reset_pts(points, boundaries, Fraction(72), Fraction(1, 1000))
        self.assertEqual(actual, {5153, 10320})
        self.assertEqual(points, [5139, 5153, 5167, 10306, 10320, 10334])
        self.assertEqual(boundaries, {371, 743})

    def test_fractional_fps_and_nonzero_start(self):
        self.assertEqual(
            utils._depth_only_reset_pts([0, 33, 67, 100], {2}, Fraction(30000, 1001), Fraction(1, 1000)),
            {67},
        )
        self.assertEqual(
            utils._depth_only_reset_pts([2000, 2014, 2028], {145}, Fraction(72), Fraction(1, 1000)),
            {2014},
        )

    def test_near_rounding_ties_and_matching_time_base(self):
        self.assertEqual(
            utils._depth_only_reset_pts([-1, 0, 1], {-1, 1}, Fraction(1), Fraction(1, 2)),
            {-1, 1},
        )
        self.assertEqual(
            utils._depth_only_reset_pts([370, 371, 372], {371}, Fraction(72), Fraction(1, 72)),
            {371},
        )

    def test_empty_boundaries_do_not_require_timing(self):
        self.assertEqual(utils._depth_only_reset_pts([None], set(), Fraction(72), None), set())

    def test_missing_timestamps_or_time_base_fail_closed(self):
        for points, time_base in (([None], Fraction(1, 1000)), ([14], None)):
            with self.subTest(points=points, time_base=time_base):
                with self.assertRaisesRegex(ValueError, "presentation timestamps"):
                    utils._depth_only_reset_pts(points, {1}, Fraction(72), time_base)


class DepthInputTest(unittest.TestCase):
    def test_order_pts_independent_arrays_and_worker_join(self):
        workers = []

        def frames():
            workers.append(threading.current_thread())
            yield from (frame(i) for i in range(7))

        with depth_input.prepared_depth_frames(frames(), 2) as prepared:
            values = list(prepared)
        self.assertEqual([pts for _, pts in values], list(range(10, 17)))
        for index, (array, _) in enumerate(values):
            np.testing.assert_array_equal(array, to_ndarray(frame(index)))
        values[-1][0].fill(0)
        self.assertTrue(np.all(values[0][0] == 32))
        self.assertIsNot(workers[0], threading.current_thread())
        self.assertFalse(workers[0].is_alive())

    def test_capacity_and_early_exit(self):
        produced = []
        reached = threading.Event()
        workers = []

        def frames():
            workers.append(threading.current_thread())
            for i in range(100):
                produced.append(i)
                if len(produced) == 3:
                    reached.set()
                yield frame(i)

        with depth_input.prepared_depth_frames(frames(), 2):
            self.assertTrue(reached.wait(5))
            self.assertEqual(len(produced), 3)  # two queued, one in flight
        self.assertFalse(workers[0].is_alive())

    def test_decode_error_reaches_caller_and_worker_joins(self):
        workers = []

        def frames():
            workers.append(threading.current_thread())
            yield frame(0)
            raise RuntimeError("decode failed")

        with self.assertRaisesRegex(RuntimeError, "decode failed"):
            with depth_input.prepared_depth_frames(frames(), 1) as prepared:
                next(prepared)
                next(prepared)
        self.assertFalse(workers[0].is_alive())

    def test_consumer_error_joins_worker(self):
        workers = []

        def frames():
            workers.append(threading.current_thread())
            yield from (frame(i) for i in range(100))

        with self.assertRaisesRegex(ValueError, "consumer"):
            with depth_input.prepared_depth_frames(frames(), 1) as prepared:
                next(prepared)
                raise ValueError("consumer")
        self.assertFalse(workers[0].is_alive())

    def test_cli_prefetch_is_opt_in(self):
        parser = utils.create_parser(required_true=False)
        self.assertEqual(parser.parse_args([]).depth_prefetch_frames, 0)
        self.assertEqual(parser.parse_args(["--depth-prefetch-frames", "24"]).depth_prefetch_frames, 24)

    def test_empty_and_invalid_capacity(self):
        with depth_input.prepared_depth_frames(iter(()), 1) as prepared:
            self.assertEqual(list(prepared), [])
        for capacity in (0, -1):
            with self.assertRaises(ValueError):
                with depth_input.prepared_depth_frames(iter(()), capacity):
                    pass


class FakeDepth:
    def __init__(self):
        self.inputs = []
        self.pts = []
        self.resets = []
        self.threads = []

    def reset(self):
        pass

    def infer_with_normalize(self, x, pts, reset_pts, **kwargs):
        self.threads.append(threading.current_thread())
        self.inputs.extend(value.clone() for value in x)
        self.pts.extend(pts)
        self.resets.append(reset_pts)
        return [value[:1] for value in x]

    def flush_with_normalize(self, **kwargs):
        return []


class DepthCallSiteTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source.mkv"
        with av.open(str(self.source), "w", format="matroska") as container:
            stream = container.add_stream("ffv1", rate=24)
            stream.width = stream.height = 8
            stream.pix_fmt = "yuv444p"
            for index in range(7):
                for packet in stream.encode(frame(index)):
                    container.mux(packet)
            for packet in stream.encode(None):
                container.mux(packet)

    def args(self, capacity):
        return SimpleNamespace(
            vf=None,
            autocrop=None,
            start_time=None,
            end_time=None,
            max_fps=24,
            state={"device": torch.device("cpu")},
            disable_amp=False,
            edge_dilation=[3, 2],
            depth_aa=False,
            batch_size=2,
            depth_prefetch_frames=capacity,
        )

    def run_depth(self, capacity, name):
        model = FakeDepth()
        output = self.root / name
        utils.process_vda_depth_only(str(self.source), str(output), model, {83}, self.args(capacity))
        with av.open(str(output)) as container:
            values = [f.to_ndarray(format="gray16le") for f in container.decode(video=0)]
        return model, values

    def test_serial_prefetch_parity_and_main_thread_inference(self):
        serial, expected = self.run_depth(0, "serial.mkv")
        candidate, actual = self.run_depth(3, "prefetch.mkv")
        self.assertEqual(len(actual), 7)
        self.assertEqual(serial.pts, candidate.pts)
        self.assertEqual(serial.resets, candidate.resets)
        self.assertTrue(all(t is threading.current_thread() for t in candidate.threads))
        for a, b in zip(serial.inputs, candidate.inputs, strict=True):
            self.assertTrue(torch.equal(a, b))
        np.testing.assert_array_equal(actual, expected)

    def test_detector_clock_matches_cut_without_changing_native_pts(self):
        with av.open(str(self.source)) as container:
            stream = container.streams.video[0]
            raw_points = [value.pts for value in container.decode(stream)]
            boundary = round(raw_points[2] * stream.time_base * stream.average_rate)
        for capacity in (0, 3):
            with self.subTest(capacity=capacity):
                model = FakeDepth()
                original = model.infer_with_normalize
                matched_indices = []

                def infer(x, pts, reset_pts, **kwargs):
                    matched_indices.extend(
                        len(model.inputs) + index for index, point in enumerate(pts) if point in reset_pts
                    )
                    return original(x, pts, reset_pts, **kwargs)

                model.infer_with_normalize = infer
                output = self.root / f"cut-{capacity}.mkv"
                utils.process_vda_depth_only(str(self.source), str(output), model, {boundary}, self.args(capacity))
                self.assertEqual(model.pts, raw_points)
                self.assertEqual(matched_indices, [2])
                with av.open(str(output)) as container:
                    self.assertEqual(len(list(container.decode(video=0))), len(raw_points))

    def test_special_or_unknown_pixel_formats_keep_serial_conversion(self):
        original_open = av.open

        class Input:
            def __init__(self, pixel_format):
                self.streams = SimpleNamespace(
                    video=[
                        SimpleNamespace(
                            average_rate=24,
                            base_rate=24,
                            codec_context=SimpleNamespace(format=pixel_format),
                        )
                    ]
                )

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def decode(self, *args):
                return iter(frame(i) for i in range(3))

        for name in (None, "nv12", "p010le"):
            pixel_format = None if name is None else SimpleNamespace(name=name)

            def open_input(filename, *args, **kwargs):
                if kwargs.get("mode") == "r":
                    return Input(pixel_format)
                return original_open(filename, *args, **kwargs)

            output = self.root / f"fallback-{name}.mkv"
            with patch.object(av, "open", side_effect=open_input):
                with patch.object(utils, "prepared_depth_frames", side_effect=AssertionError("must remain serial")):
                    model = FakeDepth()
                    utils.process_vda_depth_only(str(self.source), str(output), model, set(), self.args(24))
            self.assertEqual(len(model.inputs), 3)

    def test_boundary_mapping_failure_aborts_output_and_joins(self):
        original_open = av.open
        for capacity in (0, 3):
            for missing in ("pts", "time_base"):
                with self.subTest(capacity=capacity, missing=missing):
                    workers = []

                    class Input:
                        def __init__(self):
                            self.streams = SimpleNamespace(
                                video=[
                                    SimpleNamespace(
                                        average_rate=Fraction(24),
                                        base_rate=Fraction(24),
                                        time_base=None if missing == "time_base" else Fraction(1, 24),
                                        codec_context=SimpleNamespace(format=SimpleNamespace(name="rgb24")),
                                    )
                                ]
                            )

                        def __enter__(self):
                            return self

                        def __exit__(self, *args):
                            pass

                        def decode(self, *args):
                            workers.append(threading.current_thread())
                            for index in range(7):
                                value = frame(index)
                                if missing == "pts" and index == 1:
                                    value.pts = None
                                yield value

                    def open_input(filename, *args, **kwargs):
                        if kwargs.get("mode") == "r":
                            return Input()
                        return original_open(filename, *args, **kwargs)

                    output = self.root / f"missing-{missing}-{capacity}.mkv"
                    model = FakeDepth()
                    with patch.object(av, "open", side_effect=open_input):
                        with self.assertRaisesRegex(ValueError, "presentation timestamps"):
                            utils.process_vda_depth_only(
                                str(self.source), str(output), model, {11}, self.args(capacity)
                            )
                    self.assertEqual(model.inputs, [])
                    self.assertFalse(output.exists())
                    self.assertFalse(output.with_name("_tmp_" + output.name).exists())
                    self.assertTrue(all(not t.is_alive() for t in workers if t is not threading.current_thread()))

    def test_inference_failure_aborts_output(self):
        model = FakeDepth()
        original = model.infer_with_normalize

        def fail(*args, **kwargs):
            if model.inputs:
                raise RuntimeError("inference failed")
            return original(*args, **kwargs)

        model.infer_with_normalize = fail
        output = self.root / "failed-inference.mkv"
        with self.assertRaisesRegex(RuntimeError, "inference failed"):
            utils.process_vda_depth_only(str(self.source), str(output), model, set(), self.args(1))
        self.assertFalse(output.exists())
        self.assertFalse(output.with_name("_tmp_failed-inference.mkv").exists())

    def test_preparation_error_aborts_output_and_joins(self):
        workers = []
        calls = 0

        def fail(value):
            nonlocal calls
            calls += 1
            workers.append(threading.current_thread())
            if calls == 5:
                raise RuntimeError("RGB preparation failed")
            return to_ndarray(value)

        output = self.root / "failed.mkv"
        with patch.object(depth_input, "to_ndarray", side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, "RGB preparation failed"):
                utils.process_vda_depth_only(str(self.source), str(output), FakeDepth(), set(), self.args(1))
        self.assertFalse(output.exists())
        self.assertFalse(output.with_name("_tmp_failed.mkv").exists())
        self.assertTrue(all(not t.is_alive() for t in workers))


if __name__ == "__main__":
    unittest.main()
