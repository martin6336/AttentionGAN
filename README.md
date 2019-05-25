The code of research paper Text adversarial generation based on self-attention mechanism


## Requirement
We suggest you run the platform under Python 3.6+ with following libs:
* **TensorFlow >= 1.5.0**
* Numpy 1.12.1
* Scipy 0.19.0
* NLTK 3.2.3
* CUDA 7.5+ (Suggested for GPU speed up, not compulsory)    

Or just type `pip install -r requirements.txt` in your terminal.

## Get Started

```bash
python main.py

```

## Evaluation Results

BLEU on image COCO caption test dataset:

|       | SeqGAN | RankGAN | LeakGAN | AttentionGAN |
|-------|--------|---------|---------|--------------|
| BLEU2 | 0.853  | 0.807   | 0.918   | 0.937        |
| BLEU3 | 0.651  | 0.583   | 0.796   | 0.829        |
| BLEU4 | 0.420  | 0.373   | 0.627   | 0.664        |
| BLEU5 | 0.253  | 0.242   | 0.435   | 0.463        |

Note: this code is based on the previous work by ofirnachum. Many thanks to ofirnachum.
