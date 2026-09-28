"""Run with python -m unittest discover -s tests -v (no model downloads needed)."""
import argparse
import contextlib
import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from scripts import test_onnx


class IdentityStreamingSession:
    """Identity frame processor with the same overlap/delay contracts as exports."""

    def __init__(self, n_fft, hop_size, framed):
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.framed = framed
        self.input_size = n_fft if framed else hop_size
        self.cache = np.zeros((1, n_fft - hop_size), dtype=np.float32)
        self.frames = []

    def get_inputs(self):
        return [SimpleNamespace(name="wav_in", shape=[1, self.input_size]),
                SimpleNamespace(name="cache_in_0", shape=[1, 1])]

    def get_outputs(self):
        return [SimpleNamespace(name="wav_out", shape=[1, self.hop_size])]

    def run(self, _, inputs):
        np.testing.assert_array_equal(inputs["cache_in_0"], [[len(self.frames)]])
        chunk = inputs["wav_in"]
        assert chunk.shape == (1, self.input_size)
        frame = chunk if self.framed else np.concatenate([self.cache, chunk], axis=1)
        self.frames.append(frame.copy())
        self.cache = frame[:, self.hop_size:].copy()
        return [frame[:, :self.hop_size].copy(),
                np.array([[len(self.frames)]], dtype=np.float32)]


class WaveformFramingTests(unittest.TestCase):
    def run_audio(self, session, audio, n_fft, hop_size):
        args = argparse.Namespace(audio_path="unused.wav", onnx_path="unused.onnx",
                                  sr=16000, n_fft=n_fft, hop_size=hop_size,
                                  save_output=True)
        with patch.object(test_onnx.librosa, "load", return_value=(audio, args.sr)), \
             patch.object(test_onnx.onnxruntime, "InferenceSession", return_value=session), \
             patch.object(test_onnx.scipy.io.wavfile, "write") as write, \
             patch.object(test_onnx, "tqdm", side_effect=lambda items: items), \
             contextlib.redirect_stdout(io.StringIO()):
            test_onnx.main(args)
        return write.call_args.args[2]

    def test_legacy_and_cached_models_preserve_samples_and_alignment(self):
        for n_fft, hop in [(512, 256), (512, 160), (512, 100), (1024, 512)]:
            for length in [1, hop - 1, hop, hop + 1, 3 * hop + 17]:
                with self.subTest(n_fft=n_fft, hop=hop, length=length):
                    audio = np.linspace(0.1, 0.9, length, dtype=np.float32)
                    sessions = [IdentityStreamingSession(n_fft, hop, framed)
                                for framed in [False, True]]
                    for session in sessions:
                        result = self.run_audio(session, audio, n_fft, hop)
                        np.testing.assert_array_equal(result, audio)
                    np.testing.assert_array_equal(sessions[0].frames, sessions[1].frames)

    def test_wrong_hop_is_rejected_before_inference(self):
        session = IdentityStreamingSession(512, 256, True)
        with self.assertRaisesRegex(ValueError, "Model outputs 256"):
            self.run_audio(session, np.ones(1000, dtype=np.float32), 512, 512)
        self.assertEqual(session.frames, [])

    def test_unsupported_input_size_is_rejected(self):
        session = IdentityStreamingSession(512, 256, True)
        session.input_size = 1024
        with self.assertRaisesRegex(ValueError, "Model expects 1024"):
            self.run_audio(session, np.ones(1000, dtype=np.float32), 512, 256)
        self.assertEqual(session.frames, [])

    def test_invalid_hop_is_rejected(self):
        for hop in [0, -1, 513]:
            with self.subTest(hop=hop), self.assertRaisesRegex(ValueError, "Expected 0"):
                self.run_audio(IdentityStreamingSession(512, 256, True),
                               np.ones(1000, dtype=np.float32), 512, hop)


if __name__ == "__main__":
    unittest.main()
