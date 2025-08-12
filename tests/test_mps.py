# -*- coding: utf-8 -*-

import pytest
import torch

from supar import Parser


@pytest.mark.skipif(
    not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()),
    reason="MPS not available on this system",
)
def test_load_and_predict_on_mps():
    # Allow download if missing; avoid passing through to torch.load(**kwargs) unexpected args
    parser = Parser.load('dep-biaffine-en', device='mps', src='github')
    assert any(p.device.type == 'mps' for p in parser.model.parameters())

    # Make a simple prediction with explicit tokens to ensure non-empty length
    _ = parser.predict(['She', 'enjoys', 'playing', 'tennis', '.'], lang=None, prob=False)


