# Real or AI? A lightweight deepfake / AI-generated image detector

EECE 490/690, Introduction to Machine Learning, Project 18.

A small pretrained CNN (MobileNetV3-Large, 3.0 M parameters) is fine-tuned to tell real photographs from
AI-generated images. Besides the real / AI verdict it also names the generator the image most resembles and shows a
heat-map of where it looked. The demo screen takes two images side by side and checks both.

## What is in this folder

| Path | Contents |
|---|---|
| `detector.py` | The model, the data augmentation, prediction and the Grad-CAM heat-map (shared by the other two files) |
| `train.py` | Dataset preparation, training, evaluation and comparison of runs |
| `app.py` | The two-image demo screen |
| `Deepfake_Detector_Colab.ipynb` | The whole project as one Colab notebook (contains the three files above) |
| `data/` | The dataset: 17,000 images in `train/`, `val/`, `test/`, each with `real/` and `fake/`; see `dataset_info.json` |
| `models/` | Trained weights. `detector.pt` is the final model (12 MB) |
| `results/` | Metrics (`metrics.json`) and figures for every run, plus `comparison.md` / `comparison.png` |
| `examples/` | Six demo pairs: the same scene as a real photo and as an AI image |
| `requirements.txt` | Python libraries |

`data/_download/` (3.6 GB) is the raw download that `data/` was built from. It is not part of the submission and
can be deleted.

On GitHub the images in `data/` are left out because of their size (510 MB). `python train.py prepare` rebuilds the
same 17,000 files.

## Run the demo

```bash
source .venv/bin/activate
python app.py
```

A browser tab opens at http://127.0.0.1:7860. Drop an image into each slot (or click one of the example pairs at the
bottom) and press **Check both images**. On another computer, create the environment first with
`python -m venv .venv`, activate it, and `pip install -r requirements.txt`.

## Reproduce everything

```bash
python train.py prepare      # download Defactify (about 3.9 GB) and build data/
python train.py train        # train the final model -> models/detector.pt, results/detector/
python train.py train --no-aug --name no_aug                              # ablation: no augmentation
python train.py train --model mobilenet_v3_small --name mobilenet_v3_small # lighter backbone
python train.py train --model efficientnet_b0 --name efficientnet_b0       # optional: heavier backbone
python train.py compare      # results/comparison.md and results/comparison.png
```

**On Colab:** upload `Deepfake_Detector_Colab.ipynb`, choose `Runtime -> Change runtime type -> T4 GPU`, then
`Runtime -> Run all`. The last cell prints a public link to the demo.

## Design

**Dataset.** [Defactify](https://huggingface.co/datasets/Rajarshi-Roy-research/Defactify_Image_Dataset): real MS COCO
photographs and images generated from the same captions by Stable Diffusion 2.1, SDXL, Stable Diffusion 3, DALL-E 3
and Midjourney v6. `train.py prepare` keeps a balanced subset: 7,000 real + 7,000 AI images for training
(1,400 per generator), 500 + 500 for validation, 1,000 + 1,000 for testing. Every image, real or AI, is resized to a
256-pixel short side and saved as JPEG quality 92, so the file format cannot give the answer away.

**Model and transfer learning.** MobileNetV3-Large pretrained on ImageNet, followed by global average pooling and two
linear heads: one for *real vs AI*, one for *which of the five generators*. Training has two phases: 2 epochs with
the backbone frozen (only the heads learn), then 8 epochs fine-tuning the whole network (AdamW, one-cycle learning
rate). The checkpoint with the best validation accuracy is kept, and its decision threshold is then calibrated on
the validation set (the output bias is shifted so that the best threshold sits at 50%).

**Data augmentation.** Random crop and flip, colour jitter, and a "feed" augmentation that imitates what social media
does to a picture: random down-scaling, Gaussian blur and JPEG re-compression.

**Evaluation.** 2,000 test images never used for training or model selection. The test is repeated with the images
degraded (JPEG, blur, half size) to measure robustness, and the cost of the model is measured (parameters, GFLOPs,
CPU time per image).

## Results

Final model (`detector`: MobileNetV3-Large with augmentation), on the 2,000 test images:

| Accuracy | Precision | Recall | F1 | ROC AUC | Generator named correctly |
|---|---|---|---|---|---|
| 95.9% | 97.4% | 94.3% | 95.8% | 0.992 | 84.9% |

Correctly classified by source: real photos 97.5%, Stable Diffusion 2.1 90.5%, SDXL 94.0%, Stable Diffusion 3 90.0%,
DALL-E 3 100%, Midjourney v6 97.0%. Training took 12 minutes on a laptop GPU (Apple M5).

**What augmentation buys.** Same model, same training, with and without augmentation:

| Test images | With augmentation | Without augmentation |
|---|---|---|
| Clean | 95.9% | 97.8% |
| JPEG quality 70 | 91.8% | 88.1% |
| JPEG quality 40 | 93.2% | 84.7% |
| Blurred | 88.9% | 72.5% |
| Half size | 86.3% | 63.4% |
| Half size + JPEG 50 | 87.2% | 81.8% |

Without augmentation the model scores 2 points higher on clean files but relies on fine pixel-level traces that
disappear when the image is blurred or shrunk. With augmentation it loses 2 points on clean files and gains 16 points
on blurred images and 23 points on half-size images. Images in a real feed are never clean, so the augmented model is
the final one.

**Accuracy against computational cost.** Both trained with augmentation; CPU time measured on the laptop:

| Backbone | Test accuracy | ROC AUC | Half size + JPEG 50 | Parameters | GFLOPs | Weights | CPU time per image | Training |
|---|---|---|---|---|---|---|---|---|
| MobileNetV3-Small | 95.1% | 0.991 | 85.4% | 0.93 M | 0.05 | 3.7 MB | 29 ms | 6 min |
| MobileNetV3-Large (final) | 95.9% | 0.992 | 87.2% | 2.98 M | 0.21 | 11.9 MB | 39 ms | 12 min |

MobileNetV3-Large is the final model: it is the more accurate one, especially on degraded images. MobileNetV3-Small
comes within one point with a quarter of the compute and a third of the file size, so it is the one to pick when
computational cost matters most. EfficientNet-B0 (0.39 GFLOPs) is supported by `--model efficientnet_b0` but was not
trained for this table.

Figures: `results/detector/training_curves.png`, `results/detector/evaluation.png`, `results/comparison.png`,
`results/dataset_samples.png`, `results/augmentation_examples.png`.

## How the project meets the brief

| Requirement | Where |
|---|---|
| Transfer learning | ImageNet-pretrained backbone, frozen then fine-tuned (`train.py`, `train`) |
| Data augmentation | `train_transform` and `FeedAugment` in `detector.py`; ablation above |
| A fast, light architecture | MobileNetV3-Large, compared with MobileNetV3-Small |
| Best results at the lowest computational cost | Accuracy / parameters / GFLOPs / latency table above |
| Runs on Colab | `Deepfake_Detector_Colab.ipynb` |
| Deliverables | Code (three `.py` files + notebook), dataset (`data/`), results for the report and slides (`results/`) |

## Limitations

- The model knows five generators. Images from other or newer generators, and heavily edited images, can fool it.
- The real photos all come from MS COCO; photos from very different cameras or with heavy filters may be misjudged.
- The heat-map shows which regions weighed most in the verdict. It is an aid for discussion, not proof.
- The generator name is only meaningful for the five generators in the dataset.
