import tempfile
import unittest
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import torch

from iw3 import utils
from nunif.utils.video import OffloadFrame


class VideoDepthWriterTest(unittest.TestCase):
    def test_writes_lossless_synchronized_gray16_sidecar(self):
        values = torch.tensor(
            [[[0.0, 0.25], [0.5, 1.0]]],
            dtype=torch.float32,
        )
        expected = np.rint(values.numpy()[0] * 65535).astype(np.uint16)

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "stereo.mp4"
            writer = utils.VideoDepthWriter(str(output))
            writer.set_fps(Fraction(48, 1))
            writer.write(OffloadFrame(values, dtype=torch.uint16))
            writer.write(OffloadFrame(values, dtype=torch.uint16))
            writer.close(success=True)

            sidecar = output.with_name("stereo_depth.mkv")
            self.assertTrue(sidecar.exists())
            with av.open(str(sidecar)) as container:
                stream = container.streams.video[0]
                frames = list(container.decode(video=0))
                self.assertEqual(stream.codec_context.name, "ffv1")
                self.assertEqual(stream.average_rate, 48)
                self.assertEqual(len(frames), 2)
                for frame in frames:
                    np.testing.assert_array_equal(
                        frame.to_ndarray(format="gray16le"),
                        expected,
                    )

    def test_exact_output_path_for_depth_only_mode(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "depth.mkv"
            writer = utils.VideoDepthWriter(str(output), exact_output=True)
            writer.set_fps(48)
            writer.write(OffloadFrame(torch.zeros((1, 2, 2)), dtype=torch.uint16))
            writer.close(success=True)

            self.assertTrue(output.exists())
            self.assertFalse(output.with_name("depth_depth.mkv").exists())

    def test_failed_output_removes_temporary_sidecar(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "stereo.mp4"
            writer = utils.VideoDepthWriter(str(output))
            writer.set_fps(48)
            writer.write(
                OffloadFrame(torch.zeros((1, 2, 2)), dtype=torch.uint16)
            )
            temporary_path = Path(writer.temporary_path)
            writer.close(success=False)

            self.assertFalse(temporary_path.exists())
            self.assertFalse(output.with_name("stereo_depth.mkv").exists())


if __name__ == "__main__":
    unittest.main()
