# Samples

This directory's four text files reproduce the editing gradient from Pangram's [3.0 technical post](https://www.pangram.com/blog/pangram-3-0-technical):
- `human_ishiguro.txt` is the original human-written passage.
- `ishiguro_edit_{light,vivid,rewrite}.txt` applies the post's three editing prompts in
  increasing strength.

Local scores differ from the post, which used an unreleased 24B model.

`cx-pangram eval` uses the paper's [`pangram/editlens_iclr`](https://huggingface.co/datasets/pangram/editlens_iclr) dataset instead.
