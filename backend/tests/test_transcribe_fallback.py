from types import SimpleNamespace

import numpy as np

from app.pipeline.matching import analyze_windows
from app.pipeline.boundaries import BoundaryConfig
from app.pipeline.transcribe import Transcriber, estimate_words


class CrashyModel:
    """Mimics faster-whisper's word-timestamp IndexError on some windows."""

    def __init__(self, fail_times):
        self.fail_times = fail_times
        self.calls = []

    def transcribe(self, audio, **kw):
        self.calls.append(kw)

        def gen():
            if kw["word_timestamps"] and len(self.calls) <= self.fail_times:
                raise IndexError("boolean index did not match indexed array along axis 0; size of axis is 0 "
                                 "but size of corresponding boolean axis is 1")
            words = [SimpleNamespace(word=" what", start=1.0, end=1.2, probability=0.9),
                     SimpleNamespace(word=" the", start=1.2, end=1.3, probability=0.9),
                     SimpleNamespace(word=" fuck", start=1.3, end=1.6, probability=0.9)]
            yield SimpleNamespace(start=1.0, end=1.6, text=" what the fuck", words=words if kw["word_timestamps"] else None)

        return gen(), None


def make(fail_times):
    t = Transcriber.__new__(Transcriber)
    t.model = CrashyModel(fail_times)
    t.beam_size = 5
    return t


def test_retries_without_prompt_then_succeeds():
    t = make(1)
    segs = t.transcribe(np.zeros(16000 * 3, dtype=np.float32), "prompt")
    assert [w["word"] for w in segs[0]["words"]] == ["what", "the", "fuck"]
    assert t.model.calls[1]["initial_prompt"] is None


def test_falls_back_to_estimated_words():
    t = make(2)
    segs = t.transcribe(np.zeros(16000 * 3, dtype=np.float32), "prompt")
    words = segs[0]["words"]
    assert [w["word"] for w in words] == ["what", "the", "fuck"] and all(w["est"] for w in words)
    assert abs(words[0]["start"] - 1.0) < 1e-9 and abs(words[-1]["end"] - 1.6) < 1e-9


def test_estimated_unaligned_matches_are_not_trusted():
    words = estimate_words("what the fuck", 1.0, 1.6)
    assert analyze_windows([words], ["f_word"], [], None, BoundaryConfig()) == []
    for w in words:
        w["aligned"] = True
    assert len(analyze_windows([words], ["f_word"], [], None, BoundaryConfig())) == 1
