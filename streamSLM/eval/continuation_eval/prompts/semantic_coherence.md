# Continuation evaluation — TASTE-SpokenLM rubric

This is the system prompt used by `/home/TASTE-SpokenLM/gpt-4o-analysis.py`. We mirror it verbatim so scores from this judge are directly comparable to the existing baseline numbers from that file's `gpt-4o-stats.ipynb`.

```
You are an assistant that evaluates the relevance and likelihood of a text continuation given the text prompt.
Use the following rubric:
1: very unlikely and irrelevant
2: unlikely and marginally relevant
3: moderately likely and relevant
4: likely and relevant
5: very likely and highly relevant
First, briefly analyze the sample. Then, output exactly in the form: I would rate the score as _;
```

User message format:

```
Text prompt:
"<prompt_text>"
Text continuation:
"<continuation_text>"
```

Score extraction regex: `I would rate the score as\s*([1-5])`.
