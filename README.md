# cx-pangram

Library + CLI for AI-edit detection using the [EditLens](https://github.com/pangramlabs/EditLens) models released by Pangram Labs ([Open Pangram](https://huggingface.co/collections/pangram/open-pangram),
ICLR 2026). Given text, it returns a continuous score in `[0, 1]` quantifying the extent of
AI editing: `0` = fully human, `1` = fully AI-generated; per-segment scores are also included for longer
inputs.

## Models

| key       | checkpoint                        | base                         | gated | notes                         |
| --------- | --------------------------------- | ---------------------------- | ----- | ----------------------------- |
| `roberta` | `pangram/editlens_roberta-large`  | `FacebookAI/roberta-large`   | no    | default; ~355M        |
| `llama`   | `pangram/editlens_Llama-3.2-3B`   | `meta-llama/Llama-3.2-3B`    | yes   | qlora; needs HF token |

## Usage

```sh
cx-pangram "some text to score"
cx-pangram -p                    # read the clipboard (--paste)
wl-paste | cx-pangram            # or pipe it in
cx-pangram --file long.md --json --chunks
```

```python
from cx_pangram import EditLens

det = EditLens(model="roberta").detect(open("samples/human_ishiguro.txt").read())
print(det.score, det.band)        # 0.0x  human
```

## Notes

- Inputs are normalized as in EditLens training (`clean_text`: demojize, strip AI preambles, lowercase, collapse whitespace) so scores stay in-distribution.
- Decoding mirrors EditLens `inference.py`: `score = softmax(logits)·arange(n)/(n-1)`.
- Texts under ~50 words are flagged unreliable; longer texts are windowed (~350 words) and length-weighted into the aggregate, with the per-chunk vector retained.
- Weights are gated on HF; needs `HF_TOKEN` with granted access. On NixOS the prebuilt
  torch wheel also needs `libstdc++` + the NVIDIA driver on `LD_LIBRARY_PATH`. The included `.envrc` wires these.
