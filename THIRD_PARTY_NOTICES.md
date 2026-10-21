# Third-party notices

UTMOST includes code and weights from the following projects. These licenses apply to the listed
third-party components only; the license for UTMOST itself is TBD. The license texts are in
[`licenses/`](licenses/).

| Component              | Used in UTMOST                                                        | Source                                                                  | License                                                                             |
| ---------------------- | --------------------------------------------------------------------- | ----------------------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| UTMOS22 strong learner | Weights of the frozen side-feature model, included in the model files | [sarulab-speech/UTMOS22](https://github.com/sarulab-speech/UTMOS22)     | MIT, [`UTMOS22-MIT.txt`](licenses/UTMOS22-MIT.txt)                                  |
| SpeechMOS              | `utmost/utmos22.py`, adapted                                          | [tarepan/SpeechMOS](https://github.com/tarepan/SpeechMOS)               | MIT, [`SpeechMOS-MIT.txt`](licenses/SpeechMOS-MIT.txt)                              |
| fairseq                | wav2vec 2.0 code within SpeechMOS                                     | [facebookresearch/fairseq](https://github.com/facebookresearch/fairseq) | MIT, [`fairseq-MIT.txt`](licenses/fairseq-MIT.txt)                                  |
| Transformers           | `utmost/wavlm.py`, adapted from `models/wavlm/modeling_wavlm.py`      | [huggingface/transformers](https://github.com/huggingface/transformers) | Apache 2.0, [`transformers-Apache-2.0.txt`](licenses/transformers-Apache-2.0.txt)   |
| torchaudio             | `resample` in `utmost/audio.py`, adapted from `torchaudio.functional` | [pytorch/audio](https://github.com/pytorch/audio)                       | BSD 2-Clause, [`torchaudio-BSD-2-Clause.txt`](licenses/torchaudio-BSD-2-Clause.txt) |
| WavLM                  | Initial weights of the WavLM-Large encoder                            | [microsoft/unilm](https://github.com/microsoft/unilm)                   | MIT, [`unilm-WavLM-MIT.txt`](licenses/unilm-WavLM-MIT.txt)                          |

The adapted files state their origin and changes in their module docstrings.
