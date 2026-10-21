# UTMOST

Official implementation of **UTMOST: A Multi-Task Model for Assessing Emotional Speech Synthesis**.

UTMOST is a single model that returns quality MOS (qMOS), emotion-similarity MOS (eMOS),
valence, arousal, dominance, and the perceived emotion category of the input speech.

> **Model weights: coming soon.** The code is available now. The pretrained weights were trained
> on the VoiceMOS Challenge 2026 Track 2 dataset, whose rights are held by the challenge organizers,
> so we can release them only after the organizers make the dataset public. We will publish them on
> the Hugging Face Hub ([`sarulab-speech/UTMOST`](https://huggingface.co/sarulab-speech/UTMOST)) as
> soon as that happens. Until then, `utmost.load()` cannot download them. T02, our Track 2 submission
> to the VoiceMOS Challenge 2026, will also be released in this repository.

## Outputs

| Key                               | Output                                                                                                     |
| --------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| `qmos`                            | Quality MOS (1–5)                                                                                          |
| `emos`                            | Emotion-similarity MOS: how closely the utterance matches the requested emotion (1–5)                      |
| `valence`, `arousal`, `dominance` | Valence, arousal, and dominance (1–5)                                                                      |
| `category`                        | Probability that a listener selects each perceived emotion category (angry, happy, neutral, sad, surprise) |

The qMOS and eMOS heads predict listener-dependent scores. At inference, UTMOST averages its
predictions over 64 listener embeddings, selected at approximately equal intervals from the
embeddings of the 262 listeners in its training data.

## Model

UTMOST fine-tunes one WavLM-Large model jointly on all six outputs and augments its shared
acoustic representation with a fixed 1,025-dimensional side feature extracted by a frozen
UTMOS22-Strong model. The eMOS branch is additionally conditioned on the requested emotion and
on predicted emotion-category features. UTMOST was trained on all training ratings of the
VoiceMOS Challenge 2026 Track 2 (12,746 utterances and 262 listeners) and has 321.9M trainable
parameters.

Results on the VoiceMOS Challenge 2026 Track 2 evaluation set, as reported in the paper. Scores
are utterance-level SRCC, except CAT error, the mean absolute difference between predicted
category probabilities and listener selection proportions (lower is better).

|  qMOS |  eMOS | Valence | Arousal | Dominance | CAT error |
| ----: | ----: | ------: | ------: | --------: | --------: |
| 0.783 | 0.752 |   0.735 |   0.774 |     0.696 |     0.153 |

## Installation

```bash
pip install git+https://github.com/sarulab-speech/UTMOST
```

Python 3.9 or later and PyTorch 2.0 or later are required. On Windows, pip installs the
CPU-only build of PyTorch; to use an NVIDIA GPU, install a CUDA build first, for example
`pip install torch --index-url https://download.pytorch.org/whl/cu130`.

The model weights are downloaded from the Hugging Face Hub (`sarulab-speech/UTMOST`) on first
use. If the repository requires access, log in first with `hf auth login`, or with
`from huggingface_hub import login; login()` in Python.

## Usage

### Python

```python
import utmost

model = utmost.load()  # uses a GPU if one is available
out = model.predict("sample.wav", emotion="Happy")
print(out["qmos"], out["emos"], out["category"])

# Several files, one requested emotion per file
outs = model.predict(["a.wav", "b.wav"], emotion=["Sad", "Angry"])

# A waveform in memory
out = model.predict(waveform, sample_rate=24000, emotion="Neutral")
```

`emotion` is the emotion the synthesizer was asked to express: `Angry`, `Happy`, `Neutral`,
`Sad`, or `Surprise`. If it is omitted, UTMOST uses the category with the highest predicted
probability and reports `emotion_source: "predicted"` in the output.

With `all_emotions=True`, the output also holds eMOS, valence, arousal, and dominance for each
of the five emotions under `by_emotion`.

`utmost.load()` runs in half precision on a GPU and in single precision on a CPU. Use
`utmost.load(fp16=False)` for single precision on a GPU.

### torch.hub

```python
import torch

model = torch.hub.load("sarulab-speech/UTMOST", "utmost", trust_repo=True)
model.predict("sample.wav", emotion="Happy")
```

### Command line

```bash
utmost wavs/ --emotion happy -o scores.csv
utmost wavs/ --emotion-table emotions.txt -o scores.csv  # one "file.wav|Sad" per line
```

## Citation

```bibtex
@misc{utmost,
  title  = {{UTMOST}: A Multi-Task Model for Assessing Emotional Speech Synthesis},
  author = {Park, Joonyong and Kim, Hounsu and Park, Yoonjeong and Kawamatsu, Ryota and
            Nakata, Wataru and Saito, Yuki and Fukayama, Satoru and Nam, Juhan},
  note   = {Under review}
}
```

## License

TBD (not yet decided). Third-party code and weights included in UTMOST keep their own licenses;
see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
