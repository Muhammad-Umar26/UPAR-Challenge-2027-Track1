# assets/ (not stored in this folder)

The submitted zip (478 MB) also contains these files next to `run.py`, `metadata.yaml` and `upar_clipvit.py`
(the same module as `../../src/upar_clipvit.py`). They are left out of this folder because of their size:

| File | What it is |
|---|---|
| `model_0.pth`, `model_1.pth`, `model_2.pth` | fp16 state_dicts of the three CLIP ViT-B/16 models (about 160 MB each) |
| `config.json` | `{"H": 192, "W": 96, "mode": "pad", "attribute_names": [...40 names...], "members": [...]}`, see `config.example.json` |
| `prior.json` | training-set attribute frequencies, used only when an image cannot be read |
| `thresholds.json` | not shipped: threshold tuning gave no robust gain, so the cut stays at 0.5 |

The weights were produced with `src/upar_train.py` (commands in the top-level README) and converted to fp16 by the
packaging cell of the notebook (`notebooks/03_all_in_one.ipynb`, "package the ensemble" cell).