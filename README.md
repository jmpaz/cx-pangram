# cx-pangram

Package for AI-edit detection using the [EditLens](https://github.com/pangramlabs/EditLens) models released by Pangram Labs ([Open Pangram](https://huggingface.co/collections/pangram/open-pangram),
ICLR 2026). Given text, it returns a continuous score in `[0, 1]` quantifying the extent of
AI editing: `0` = fully human, `1` = fully AI-generated; per-segment scores are also included for longer
inputs.

## Models

| key       | checkpoint                        | default base                | notes                                       |
| --------- | --------------------------------- | --------------------------- | ------------------------------------------- |
| `llama`   | `pangram/editlens_Llama-3.2-3B`   | `unsloth/Llama-3.2-3B`      | qlora 3B; default where an accelerator exists |
| `roberta` | `pangram/editlens_roberta-large`  | `FacebookAI/roberta-large`  | ~355M; default on CPU-only hosts            |

Both checkpoints are gated on Hugging Face, so scoring needs an `HF_TOKEN` with granted access.
`llama`'s default base is the ungated `unsloth/Llama-3.2-3B` mirror (weight-identical to the
gated `meta-llama/Llama-3.2-3B`); override either with `--base`.

The default model is chosen by device: `llama` when CUDA or Apple-Silicon MPS is available,
`roberta` on CPU-only hosts.

## Devices and extras

Device is auto-selected `cuda > mps > cpu`; override with `--device cuda|mps|cpu`. The `llama`
backbone loads 4-bit (bitsandbytes) on CUDA with the `[cuda]` extra present; everywhere
else it loads unquantized in bf16.

- base: scores raw text and the `roberta` / unquantized-`llama` paths.
- `[cuda]`: adds bitsandbytes for 4-bit `llama` on NVIDIA.
- `[contextualize]`: adds the `contextualize` resolver so refs (URLs, Are.na blocks/channels, [etc.](https://github.com/jmpaz/cx-plugins)) can be scored.

## Usage

```sh
cx-pangram --text "some text to score"        # raw text
echo "some text" | cx-pangram                 # or via stdin

cx-pangram https://example.com/post ...        # resolve refs through contextualize and score (needs [contextualize])
cx-pangram --model llama --device cpu --json ...
```

```python
from cx_pangram import EditLens

det = EditLens().detect(open("samples/human_ishiguro.txt").read())  # auto device + model
print(det.score, det.band, det.model, det.calibrated)
```

## Notes

- Inputs are normalized as in EditLens training (`clean_text`: demojize, strip AI preambles, lowercase, collapse whitespace) so scores stay in-distribution.
- Decoding mirrors EditLens `inference.py`: `score = softmax(logits)·arange(n)/(n-1)`.
- Texts under ~50 words are flagged unreliable; longer texts are windowed (~350 words) and length-weighted into the aggregate, with the per-chunk vector retained.
- On NixOS the prebuilt torch wheel needs `libstdc++` on `LD_LIBRARY_PATH` (plus the NVIDIA driver path on CUDA hosts); the included `.envrc` wires these.
