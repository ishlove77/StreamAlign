#!/usr/bin/env python3
"""Plain-RVQ variant: patches the model import, runs original training."""

import os
import sys

_STREAMASR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _STREAMASR_ROOT)

from models.model_tokenizer import Data2VecSemanticAcousticModel

import train.train_tokenizer_base as _orig

_orig.Data2VecSemanticAcousticModel = Data2VecSemanticAcousticModel

if __name__ == "__main__":
    import multiprocessing as mp
    mp.set_start_method("spawn", force=True)
    _orig.main()
