# samples

- `human_ishiguro.txt`: a fully human-written paragraph published in the [Pangram 3.0 technical post](https://www.pangram.com/blog/pangram-3-0-technical), used there as the source text for their AI-edit gradient demonstration.
- `ishiguro_edit_{light,vivid,rewrite}.txt`: reproductions of that demonstration: the paragraph above, progressively AI-edited under the post's three prompts ("Clean this up, I'm trying to submit my paper to a literary journal", "Make the language more vibrant", "Rewrite this in the style of Ishiguro").Absolute values differ from the blog, which scores with an unreleased 24B model.

The paper's actual evaluation set is the [`pangram/editlens_iclr`](https://huggingface.co/datasets/pangram/editlens_iclr) dataset.
