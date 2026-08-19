"""Streaming-SLM evaluation suite: SALMon (acoustic) + StoryCloze (semantic).

The evaluation mirrors TASTE-SpokenLM's protocol (see
``/home/TASTE-SpokenLM/{salmon,storycloze}.py``) but
plugs in StreamSLM's two-stream (text + RVQ acoustic) log-likelihood:

    score(wav, text) = -(text_NLL + acoustic_NLL)        # higher = better

For each (positive, negative) pair we count the prediction correct iff
``pos_score > neg_score``.

Per the StreamSLM training spec we deliberately drop the duration head from
the score (regression L1 / MSE has no meaningful log-likelihood).

Components
----------
- ``extractor.RVQUnitExtractor``  raw wav (+ ASR transcript) -> SubwordUnits
  via the StreamAlign teacher (constrained-RNNT alignment with a single
  whole-utterance interval, since SALMon / StoryCloze ship no TextGrids).
- ``scorer.StreamSLMScorer``       SubwordUnits -> joint NLL using the
  trained StreamSLM checkpoint (text CE + per-codebook acoustic CE, summed).
- ``salmon.py`` / ``storycloze.py`` benchmark drivers.
"""
