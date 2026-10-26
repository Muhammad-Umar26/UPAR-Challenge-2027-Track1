# UPAR Challenge 2027, Track 1: Pedestrian Attribute Recognition

Team: Muhammad Umar (23K-0023), Raghib Rizwan Rabbani (23K-0012), Syed Fahad Faheem Shah (23K-0062)

This folder holds everything we used for the course assignment on the UPAR Challenge 2027 (Real-World
Surveillance workshop at WACV 2027, Codabench competition 18180, Track 1). The task is to predict 40 binary
attributes (age, gender, hair length, clothing length and colour, accessories) for each pedestrian crop. The
ranking metric is Challenge_Avg, the harmonic mean of mA and instance-level F1, scored on a hidden test set from an
unseen domain.

**Final result:** Challenge_Avg 0.7773 (mA 0.7754, Instance_F1 0.7792), submission ID 1010210, rank 42 on the
leaderboard. This is an average of two CLIP ViT-L/14 models and it was our last allowed submission. Our best single model
(one CLIP ViT-L/14, ID 1006798) scored 0.7788, and the earlier three-model CLIP ViT-B/16 ensemble scored 0.7694.
The top score at that time was 0.8021.

## What is in here

| Path | Contents |
|---|---|
| `src/` | The code we trained with: image cache builder, GPU data pipeline, training script, CLIP model, metric |
| `codabench_package/` | What we uploaded to Codabench: `run.py`, `metadata.yaml`, the plain-PyTorch CLIP module (`../src/upar_clipvit.py`) and a description of the weight files |
| `results/` | Held-out-source ablations and the leaderboard progress as CSV |
| `report/` | The written report (`report.pdf`, 3 pages) and the proof screenshots in `figures/` |
| `demo/` | The class presentation (`UPAR_Challenge_Presentation.pptx`, 7 slides with speaker notes, about 5 minutes) |
| `notebooks/all_in_one.ipynb` | The Colab notebook, cleaned and put in run order (setup and data, analysis, cache, training, experiments, Codabench packages for B/16 and L/14), with the outputs we kept |

The weights (about 0.5 GB per submission zip), the datasets and the image cache are not stored here because of their size.

## Method in five lines

1. Every crop is resized to 192x96 keeping its aspect ratio and padded with the ImageNet mean colour, then cached
   once as uint8 (`src/upar_preproc.py`, `src/upar_cache.py`).
2. Augmentation runs on the GPU (`src/upar_data.py`): flip, small affine, brightness/contrast/saturation,
   low-resolution simulation and random erasing. No hue changes, because colours are labels.
3. Loss: one sigmoid per attribute, binary cross-entropy with positive weights capped at 10, the loss masked where a
   source never annotates an attribute, and source-balanced sampling (share proportional to size to the power 0.5).
4. Backbone: CLIP with layer-wise learning-rate decay (`src/upar_clipvit.py`, `src/upar_train.py`). We started with
   ConvNeXt-Tiny, moved to CLIP ViT-B/16 after leave-one-source-out experiments, and finally to ViT-L/14 (196x98 input).
5. Final model: two CLIP ViT-L/14 models trained on train+val (plain fine-tuning and linear-probe-then-fine-tune),
   averaged in logit space, cut at 0.5. One L/14 alone scored slightly higher (0.7788 against 0.7773), which is within
   noise; the ensemble was our last allowed submission, so it is the final score.

| Submission | Model | Challenge_Avg |
|---|---|---|
| 1 | Sample baseline | 0.4955 |
| 2 | ConvNeXt-Tiny | 0.7288 |
| 3 | CLIP ViT-B/16, one model | 0.7668 |
| 4 | CLIP ViT-B/16, three models | 0.7694 |
| 5 | CLIP ViT-L/14, one model (best single) | 0.7788 |
| 6 | CLIP ViT-L/14, two models (final) | 0.7773 |

## How to reproduce (Google Colab, free T4)

The scripts assume the Colab layout `/content/drive/MyDrive/upar/...` (code in `upar/code`, runs in `upar/runs`).

1. Download the organisers' repository (github.com/speckean/UPAR-Challenge-2027, it has `download_datasets.py`,
   `peta_file_mapping.txt` and the annotation CSVs), zip it, put the zip on Drive, and run the data setup cells of
   `notebooks/03_all_in_one.ipynb` (downloads Market1501, PA-100K and PETA and verifies every image).
2. Copy `src/*.py` to `MyDrive/upar/code/`, then build the cache (`upar_cache.build_split`, about two minutes).
3. Install `open_clip_torch` (training only) and train. The two models of the final submission (L/14, 196x98 input):

```
python upar_train.py --name final_l14_a    --final --arch clip_l14 --clip-size 196x98 --batch-size 64 --balance 0.5 --epochs 3 --warmup-epochs 0.5 --lr 1e-5 --llrd 0.85 --lr-head 1e-3
python upar_train.py --name final_l14_lpft --final --arch clip_l14 --clip-size 196x98 --batch-size 64 --balance 0.5 --epochs 4 --warmup-epochs 0.5 --lr 1e-5 --llrd 0.85 --lr-head 1e-3 --freeze-epochs 1
```

   Each takes about 2 to 2.5 hours on a T4 (about 37 minutes per epoch, 12.9 GB peak memory). The earlier B/16
   models (submissions 3 and 4) were trained like this, about 58 minutes each:

```
python upar_train.py --name final_clip_b    --final --arch clip_b16 --balance 0.5 --epochs 5 --lr 1e-5 --llrd 0.7  --lr-head 1e-3
python upar_train.py --name final_clip_a    --final --arch clip_b16 --balance 0.5 --epochs 5 --lr 3e-5 --llrd 0.75 --lr-head 1e-3
python upar_train.py --name final_clip_b_s1 --final --arch clip_b16 --balance 0.5 --epochs 5 --lr 1e-5 --llrd 0.7  --lr-head 1e-3 --seed 1
```

   Experiments used `--holdout PETA` (or `Market1501`) instead of `--final` to train on two sources and score the
   third with the official metric (`src/upar_eval.py`).
4. Package: copy `codabench_package/` plus the state_dicts as `assets/model_{0,1}.pth` (int8 for the two-model zip, fp16
   for one model), write `assets/config.json` (see `config.example.json`) and `assets/prior.json`, and zip the contents
   (not the folder). The packaging cell (Part 7 of the notebook) also runs `run.py` on 500 validation images without
   `open_clip` or `timm` and compares the result with the training pipeline before you upload.

## Rules we followed

- Only the organisers' training and validation data were used for training. No other attribute data.
- Pretrained weights: ImageNet (torchvision ConvNeXt, ResNet) and OpenAI CLIP, a foundation model that was not trained for
  attribute recognition.
- No test image was used for training, pseudo-labelling or setting hyper-parameters. Recipes were chosen on held-out
  validation sources; the leaderboard was only used to compare finished candidates.
- `run.py` uses the same models and threshold for every image and never looks at `sample['image']` or the image folder.
- Inference needs only torch, torchvision, numpy and Pillow (the CLIP tower is shipped as plain PyTorch).

## Known limits

- Most ablations are single-seed, so differences below about 0.005 should be read as noise. The L/14 runs had no
  held-out check at all (they were trained on train+val only, because of the time limit), so their settings were taken
  from the B/16 experiments.
- Grader time: one L/14 with flip TTA took 141 s in total on Codabench (limit 5 minutes). `run.py` drops flip TTA by
  itself when the projected time is too long, which is what protects the two-model zip.
- `auto_valid` in `src/upar_eval.py` is our approximation of the official "invalid attribute" exclusion.
- The scripts were written for Colab and are not meant to run unchanged on other machines.