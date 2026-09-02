"""`StreamingScorer` must reproduce offline scoring exactly, in bounded memory.

An online scorer that merely *approximates* the offline one would make every reported AUC
unverifiable, so these tests pin equality rather than closeness, and separately pin the
property that pays for it: the model never sees more than a bounded window at a time.
"""

import pytest
import torch

from src.models.ur_dmu.configuration_ur_dmu import URDMUConfig
from src.models.ur_dmu.modeling_ur_dmu import URDMUForVideoAnomalyDetection
from src.modules import translayer
from src.streaming import StreamingScorer, receptive_field


@pytest.fixture(autouse=True)
def _restore_impl():
    saved = (translayer._ATTN_IMPL, translayer._ATTN_WINDOW, translayer._ATTN_CHUNK)
    yield
    translayer._ATTN_IMPL, translayer._ATTN_WINDOW, translayer._ATTN_CHUNK = saved


def _model(dim=64, layers=2):
    torch.manual_seed(0)
    cfg = URDMUConfig(feature_size=dim, hidden_size=64, num_layers=layers, num_heads=2, mem_size=16)
    return URDMUForVideoAnomalyDetection(cfg).eval()


def _causal(window=8):
    translayer._ATTN_IMPL, translayer._ATTN_WINDOW = "causal", window


def test_receptive_field_refuses_unbounded_attention():
    translayer._ATTN_IMPL = "eager"
    with pytest.raises(ValueError, match="bounded attention span"):
        receptive_field(_model())


def test_receptive_field_matches_the_architecture():
    _causal(window=8)
    model = _model(layers=2)
    left, right = receptive_field(model)
    # 2 translayer layers x window 8, backwards only, plus the k=3/pad=1 embedding conv
    assert (left, right) == (2 * 8 + 1, 1)


def test_streaming_reproduces_full_sequence_scoring():
    _causal(window=8)
    model = _model()
    feat = torch.randn(2, 200, 64)

    with torch.no_grad():
        offline = torch.stack(
            [model(video=feat[None, c : c + 1]).scores.squeeze(0).squeeze(-1) for c in range(2)]
        ).mean(0)
    online = StreamingScorer(model, chunk=32).score(feat)

    assert online.shape == offline.shape
    assert torch.allclose(online, offline, atol=1e-5), (online - offline).abs().max()


def test_streaming_is_chunk_size_invariant():
    _causal(window=8)
    model = _model()
    feat = torch.randn(1, 137, 64)

    a = StreamingScorer(model, chunk=16).score(feat)
    b = StreamingScorer(model, chunk=64).score(feat)
    assert torch.allclose(a, b, atol=1e-5), (a - b).abs().max()


def test_block_stream_matches_the_chunked_score():
    _causal(window=8)
    model = _model()
    feat = torch.randn(1, 120, 64)

    scorer = StreamingScorer(model, chunk=20)
    blocks = [feat[:, i : i + 20] for i in range(0, 120, 20)]
    streamed = torch.cat(list(scorer.stream(blocks)))

    assert torch.allclose(streamed, scorer.score(feat), atol=1e-5)


def test_the_model_never_sees_more_than_the_bounded_window():
    """This is the whole point: cost per chunk must not grow with the video length."""
    _causal(window=8)
    model = _model()
    scorer = StreamingScorer(model, chunk=32)

    seen = []
    inner = scorer._forward
    scorer._forward = lambda x: (seen.append(x.shape[1]), inner(x))[1]

    scorer.score(torch.randn(1, 1000, 64))
    assert max(seen) <= scorer.left + scorer.chunk + scorer.right
    assert max(seen) < 1000
