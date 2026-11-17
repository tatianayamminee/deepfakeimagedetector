# Run comparison

'Feed accuracy' = test accuracy after feed (half size + JPEG 50).

| Run | Backbone | Augmentation | Accuracy | AUC | Feed accuracy | Params (M) | GFLOPs | CPU ms/img | Train min |
|---|---|---|---|---|---|---|---|---|---|
| detector | mobilenet_v3_large | yes | 95.90% | 0.9918 | 87.20% | 2.98 | 0.21 | 38.9 | 12.3 |
| mobilenet_v3_small | mobilenet_v3_small | yes | 95.10% | 0.9906 | 85.40% | 0.93 | 0.05 | 28.7 | 6.1 |
| no_aug | mobilenet_v3_large | no | 97.80% | 0.9976 | 81.85% | 2.98 | 0.21 | 39.5 | 13.2 |
