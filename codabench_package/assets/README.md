# Weight files and configuration (not stored in this repository)

The final submission (`06_submission_l14_ensemble.zip`, Codabench ID 1010210) puts these files into `assets/` next to `run.py`:

| File | What it is |
|---|---|
| `model_0.pth`, `model_1.pth` | state_dicts of the two CLIP ViT-L/14 models, int8 weights (per-row scales stored as `<name>::q` and `<name>::s`), about 305 MB each |
| `config.json` | `{"H": 192, "W": 96, "mode": "pad", "attribute_names": [...40 names...], "members": [...]}`, see `config.example.json` |
| `prior.json` | training-set attribute frequencies, used only when an image cannot be read |
| `thresholds.json` | not shipped: threshold tuning gave no robust gain, so the cut stays at 0.5 |

The best single model (`submission_l14.zip`, ID 1006798, 0.7788) is the same layout with only `final_l14_a`, stored as fp16 (`INT8 = False`).
The earlier B/16 package (`05_clip_ens.zip`, three fp16 models, 0.7694) used `arch: "clip_b16"` members.

The weights were produced with `src/upar_train.py` (commands in the top-level README) and converted by the packaging cell of the
notebook (`notebooks/all_in_one.ipynb`, Part 7). The zip also holds `run.py`, `metadata.yaml` and `upar_clipvit.py`.
