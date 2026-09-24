# QraXAi

> Create small open-source models. But very small.

QraXAi is an open project focused on training compact, GPT-style language models
from scratch: tokenization, model code, training loop and a Hugging Face-ready
export — small enough to read end to end and reproducible on a single consumer GPU.

[![Model](https://img.shields.io/badge/Model-QraXAi--Basic--32M-orange.svg)](https://huggingface.co/coderian/QraXAi-Basic-32M)
[![Collection](https://img.shields.io/badge/Hugging%20Face-Collection-yellow.svg)](https://huggingface.co/collections/coderian/qraxai)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://huggingface.co/coderian/QraXAi-Basic-32M/blob/main/LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-3776ab.svg?logo=python&logoColor=white)](https://www.python.org/)

## What we do

- Train decoder-only transformers from scratch with Hugging Face `transformers`
  and the GPT-2 BPE tokenizer
- Keep the code minimal and readable: `model.py`, `configuration_qraxai.py`,
  `train.py`, `push.py`
- Export every model as a self-contained folder (`config.json` with `auto_map`,
  weights, tokenizer, model code) so it loads with `trust_remote_code=True`
- Keep training reproducible on modest hardware — the first model was trained on
  an 8 GB laptop GPU, about 21 minutes per epoch

## Projects

| Repository | Description | Links |
|---|---|---|
| `QraxAi-Basic-32M` | Reference implementation and first release: a 32.1M parameter GPT-style model plus the full training pipeline | [GitHub](https://github.com/QraXAi/QraxAi-Basic-32M) · [Hugging Face](https://huggingface.co/coderian/QraXAi-Basic-32M) |

## Model at a glance

| | |
|---|---|
| Parameters | 32.1M (fp32) |
| Layers / hidden / heads | 8 / 256 / 8 |
| Context length | 256 tokens |
| Tokenizer | GPT-2 BPE, vocabulary 50,257 |
| Training data | `exnivo/tinybrain-pretrain-corpus-2b`, first 75,000 rows (~61M tokens) |
| Training | 1 epoch, bf16 autocast, AdamW 3e-4, RTX 4060 Laptop |
| Loss | 10.85 (init) to 6.02 |
| License | Apache-2.0 |

## Quickstart

Train your own:

```bash
git clone https://github.com/QraXAi/QraxAi-Basic-32M.git
cd QraxAi-Basic-32M
python -m venv venv && source venv/bin/activate
pip install torch transformers huggingface_hub datasets

python dataset/download.py    # ~284 MB English corpus -> dataset/data.txt
python train.py               # writes the HF-ready folder to qraxai/
python push.py                # set REPO_ID first, then upload
```

Use the released model:

```python
from transformers import AutoModelForCausalLM, GPT2TokenizerFast

model_id = "coderian/QraXAi-Basic-32M"
tokenizer = GPT2TokenizerFast.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True).eval()

inputs = tokenizer("The purpose of this experiment is", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=64, do_sample=True,
                         temperature=0.8, top_k=50, pad_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(outputs[0]))
```

## Roadmap

- [x] GPT-style model from scratch
- [x] Readable training loop (bf16 + batched tokenization)
- [x] Hugging Face-ready export and push script
- [ ] LR schedule: warmup + cosine decay
- [ ] Longer training / larger model
- [ ] Validation split and evaluation
- [ ] KV cache for faster generation

## Contributing

Issues and pull requests are welcome in any of the repositories above. For model
issues, please mention the model id (`coderian/QraXAi-Basic-32M`) and the prompt
you used.

## License

Model artifacts: Apache-2.0. Tokenizer files come from OpenAI's GPT-2 release (MIT).
The training corpus keeps its own terms; check them before redistributing
derivatives.
