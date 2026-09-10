# Third-party notices

## OpenADMET PXR challenge data

Files under `data/openadmet/` originate from the
[OpenADMET PXR train/test dataset](https://huggingface.co/datasets/openadmet/pxr-challenge-train-test).
The frozen curation, prepared-state, feature, and prediction tables under
`data/derived/` and `data/published/` are derived from those public records.
The upstream dataset is licensed under Apache License 2.0; a copy is included
at `licenses/APACHE-2.0.txt`.

## Nesso-1

The cached representations under `data/published/` were generated with the
[Nesso-1 model](https://huggingface.co/recursionpharma/nesso) and the
[Nesso-1 source code](https://github.com/recursionpharma/nesso), both released
under Apache License 2.0. The Nesso checkpoint itself is not redistributed in
this repository; `scripts/download_nesso_checkpoint.py` retrieves it directly
from the upstream model repository.
