# JurassiClass: Dinosaur Species Classification with Convolutional Neural Networks

This repository documents a small deep-learning project focused on classifying dinosaur species from images. The work covers dataset construction, automated and manual data cleaning, model design, transfer learning, hyperparameter search, and a short analysis of the learned latent spaces.

## Dataset

The dataset was built by scraping images for eight dinosaur species: Microceratus, Pachycephalosaurus, Parasaurolophus, Spinosaurus, Stegosaurus, Triceratops, Tyrannosaurus, and Velociraptor. Scraping was performed using simple query-based collection procedures, followed by a structured cleaning pipeline. Each downloaded image was first processed using a CLIP-based cosine-similarity filter that compared the embedding of the image with text prompts describing the target species. This removed a substantial portion of unrelated pictures, such as modern animals, landscapes, and various non-dinosaur artifacts.

A manual curation phase followed, during which unrealistic cartoons, toys, or stylized illustrations were removed unless they preserved plausible anatomical structure. Realistic paleoart, CGI renders, and detailed reconstructions were generally kept. This created a dataset with high visual variability, partly due to differences in artistic style and partly due to the evolving paleontological understanding of each species. For example, older representations of Velociraptors differ strongly from more modern, feathered reconstructions. This variability introduces label noise, but also provides a more challenging and realistic classification task.

The dataset is used with stratified cross-validation. Instead of relying on a single fixed train/test split, each experiment reports performance on held-out folds. In the standard setup, a 5-fold stratified split is used: models are trained on 4 folds and evaluated on the remaining fold, repeated across all folds. Checkpoints are selected by minimum validation loss within each run to keep selection consistent across models.

### Example Images

The following figure shows a sample of curated images across several species:

![Dataset Examples](images/dataset_examples.png)

## Models

Two convolutional neural networks were implemented using PyTorch Lightning: **RexNet**, which uses a ResNet-34 backbone, and **EfficentRex**, which uses EfficientNetV2-S. Although both have roughly 20 million parameters, they represent different architectural philosophies. ResNet relies on residual blocks with uniform channel sizes, while EfficientNetV2-S uses compound scaling, depthwise convolutions, and inverted bottleneck layers, typically achieving a more favorable accuracy–efficiency balance.

Both networks were pretrained on ImageNet and adapted for eight-class classification. Training utilities, preprocessing functions, and augmentation steps were defined through auxiliary modules included in the repository. Cross-validation and grid-search logic were handled by dedicated scripts, allowing systematic experimentation across different hyperparameters.

## Experiments

The first set of experiments retrained only the final classifier layer while keeping all pretrained backbone weights frozen. Using a classifier learning rate of 0.005, EfficientNetV2-S reached a mean accuracy of **0.8247** (std 0.0102) across five folds, while ResNet-34 achieved **0.7278** (std 0.0078). Even without fine-tuning, EfficientNet provided notably stronger features for this task.

The second set of experiments fine-tuned the highest layers of each network along with the classifier. Since high-level ImageNet features are not necessarily optimal for distinguishing dinosaur species, adapting these later layers produced a significant improvement. With a classifier learning rate of 0.005 and a backbone learning rate of 0.0005, EfficientNet achieved **0.9165** accuracy (std 0.007), while ResNet reached **0.8525** (std 0.0250). The effect of fine-tuning was substantial for both models, and EfficientNet again showed superior performance and stability.

A third experiment introduced weight decay (0.001) using AdamW. This raised the EfficientNet accuracy to **0.9228** (std 0.0086). ResNet, however, decreased to **0.8418** (std 0.0331). At this point, the stronger overall performance of EfficientNet justified focusing exclusively on this architecture for the hyperparameter search.

## Hyperparameter Search

To refine training further, a nested cross-validation procedure was used, consisting of two outer folds and two inner folds (kept small for compute reasons). The outer loop defines the held-out test split for evaluation (roughly 80/20), while the inner loop is used to select hyperparameters by training on a subset of the outer training split and validating on another held-out subset (again roughly 80/20 within the outer training data). This structure reduces the bias that can arise when hyperparameters are tuned and evaluated on the same data.

### Learning Rate Search (Nested CV)

The first search tested two learning rates for the classifier layer (0.003 and 0.005) and two scaling factors for the backbone (reductions by 5 or 10). The best-performing configuration was a classifier learning rate of **0.003** with a backbone learning-rate reduction factor of **10**, producing the following aggregated results:

| Classifier LR | Factor | Mean Accuracy | Std    |
| ------------- | ------ | ------------- | ------ |
| 0.003         | 10     | **0.9141**    | 0.0178 |
| 0.003         | 5      | 0.9148        | 0.0267 |
| 0.005         | 10     | 0.9038        | 0.0278 |
| 0.005         | 5      | 0.9018        | 0.0141 |

### Weight Decay Search (Nested CV)

Using the best learning-rate configuration, a second search varied the weight decay of the classifier layer (wd1) and backbone layers (wd2). The strongest configuration used **wd1 = 1e-4** and **wd2 = 1e-4**, yielding:

| wd1  | wd2  | Mean Accuracy | Std    |
| ---- | ---- | ------------- | ------ |
| 1e-4 | 1e-4 | **0.9203**    | 0.0118 |
| 1e-4 | 1e-5 | 0.9087        | 0.0109 |
| 1e-3 | 1e-4 | 0.8990        | 0.0193 |
| 1e-3 | 1e-5 | 0.9162        | 0.0031 |

Across all experiments, EfficientNetV2-S proved more accurate, more stable, and more responsive to fine-tuning and regularization than ResNet-34. Fine-tuning the last block, introducing moderate weight decay, and using a relatively small classifier learning rate with a strong reduction for the backbone all contributed to improved performance. These results indicate that EfficientNet can adapt its high-level representation to the highly variable visual domain of dinosaur imagery, despite substantial stylistic diversity and label uncertainty within the dataset.

## Latent Space Exploration

To better understand what the models learn, I also analyzed the latent representations produced by each network on the held-out data. For each trained checkpoint, I extracted the feature vector immediately before the classification layer (the “penultimate” representation) for all samples in the evaluation split. These vectors were then projected into two dimensions using UMAP to obtain a readable visualization of the feature space. In addition, I ran K-means on the latent vectors and computed a “K-means accuracy”, which measures how well the unsupervised clusters align with the true species labels.

The visualizations were generated as interactive Altair plots and exported as HTML. Each plot supports coloring points either by the true label (species) or by the K-means cluster assignment, and it is possible to switch between the five checkpoints corresponding to the five cross-validation folds. The plots also report, for each checkpoint, the test accuracy and the K-means accuracy, so it is easy to relate classification performance to how well the latent space separates the classes.

In the classifier-only setting (frozen backbone), the latent space is expectedly messy for both architectures. Since the feature extractor is largely fixed (ImageNet features), the model can only reshape the decision boundary in the last layer, but the underlying embedding is not optimized to separate dinosaur species. Consistently, K-means accuracy is low (around **30–35%**), and clusters overlap substantially.

When the last block is fine-tuned, the situation changes: the embeddings become much more class-separable, and K-means accuracy rises dramatically. In these runs, the ResNet latent space typically reaches around **80–85%** K-means accuracy, while EfficientNet reaches around **~90%**, indicating a more cleanly clustered representation. This matches the supervised results, where EfficientNet also achieves higher test accuracy.

Finally, I produced an additional interactive plot where hovering a point shows the corresponding image. This makes it easy to inspect ambiguous examples, mislabeled data, and failure cases, and it highlights that some errors are driven by noisy images or uncertain reconstructions rather than purely model limitations.

### Interactive Plots

These are interactive Altair HTML exports (UMAP projection of penultimate-layer embeddings + optional K-means clustering).

**Classifier-only (frozen backbone):**

* EfficientNet: https://nicogreeco.github.io/JurassicClass/images/latent/classifeir_only_EfficentRex.html
* ResNet: https://nicogreeco.github.io/JurassicClass/images/latent/classifeir_only_RexNet.html

**Last-block fine-tuning:**

* EfficientNet: https://nicogreeco.github.io/JurassicClass/images/latent/last_block_finetune_EfficentRex.html
* ResNet: https://nicogreeco.github.io/JurassicClass/images/latent/last_block_finetune_RexNet.html

**Hover-to-preview images:**

* EfficientNet: https://nicogreeco.github.io/JurassicClass/images/latent/last_block_finetune_EfficentRex_images.html
* ResNet: https://nicogreeco.github.io/JurassicClass/images/latent/last_block_finetune_RexNet_images.html

## LoRA vs Full Fine-Tuning on Larger Backbones (RexNet-101 & ViTRex-B)

To study **parameter-efficient fine-tuning**, I ran a new set of 5-fold stratified cross-validation experiments comparing **Full Fine-Tuning (FFT)** against **LoRA** (Low-Rank Adaptation). LoRA injects small trainable low-rank updates into existing layers (instead of updating all backbone weights), typically reducing the number of trainable parameters while aiming to preserve most of the pretrained representation.

I initially planned to include **EfficientNetV2-S**, but the LoRA implementation I used had issues with **EfficientNet’s depthwise convolutions**. For this reason, I focused on two large, standard backbones: **RexNet (ResNet-101 based)** and **ViTRex (ViT-B/16 based)**.

### Experimental setup
For each backbone I tested:
- **FFT** vs **LoRA ranks** *r = 10, 50, 100*
- Two training scopes:
  - **full**: adapt the whole backbone (FFT trains all backbone weights; LoRA trains adapters across the backbone)
  - **partial**: adapt only late layers (ResNet: last block / `layer4`; ViT: encoder blocks **8–11**). In LoRA-partial, only adapters inside the unfrozen layers are trained.

**Metrics tracked** (averaged across folds): validation loss/accuracy, test accuracy, **peak GPU memory** (PyTorch peak allocated), **avg epoch time**, and **epochs trained** (with early stopping; checkpoint chosen by minimum validation loss).

### Results (mean ± std across 5 folds)

#### ViTRex (ViT-B/16 backbone)

| Scope   | Method     | Test acc    | Val loss        | Peak GPU mem (MB)   | Avg epoch time (s)   | Epochs trained   |
|:--------|:-----------|:------------|:----------------|:--------------------|:---------------------|:-----------------|
| full    | FFT        | 0.90 ± 0.01 | 0.3101 ± 0.0606 | 9065.96 ± 0.00      | 54.74 ± 0.35         | 29.60 ± 9.29     |
| full    | LoRA r=10  | 0.86 ± 0.01 | 0.3846 ± 0.0532 | 8267.76 ± 0.00      | 48.11 ± 0.33         | 25.40 ± 3.21     |
| full    | LoRA r=50  | 0.88 ± 0.01 | 0.3383 ± 0.0541 | 8418.04 ± 0.00      | 48.96 ± 0.32         | 23.20 ± 2.05     |
| full    | LoRA r=100 | 0.89 ± 0.00 | 0.3240 ± 0.0706 | 8630.73 ± 0.99      | 50.95 ± 0.51         | 25.60 ± 4.16     |
| partial | FFT        | 0.89 ± 0.00 | 0.3321 ± 0.0664 | 3409.05 ± 0.00      | 77.12 ± 0.64         | 25.20 ± 4.92     |
| partial | LoRA r=10  | 0.84 ± 0.03 | 0.4682 ± 0.0553 | 6176.01 ± 0.00      | 82.55 ± 0.51         | 28.20 ± 5.81     |
| partial | LoRA r=50  | 0.87 ± 0.01 | 0.3971 ± 0.0658 | 6241.27 ± 0.00      | 79.09 ± 7.03         | 25.60 ± 5.94     |
| partial | LoRA r=100 | 0.88 ± 0.02 | 0.3648 ± 0.0567 | 6340.17 ± 0.00      | 69.67 ± 0.82         | 27.60 ± 6.73     |

#### RexNet (ResNet-101 backbone)

| Scope   | Method     | Test acc    | Val loss        | Peak GPU mem (MB)   | Avg epoch time (s)   | Epochs trained   |
|:--------|:-----------|:------------|:----------------|:--------------------|:---------------------|:-----------------|
| full    | FFT        | 0.91 ± 0.02 | 0.2896 ± 0.0410 | 8480.29 ± 1.05      | 73.06 ± 1.02         | 23.20 ± 4.44     |
| full    | LoRA r=10  | 0.78 ± 0.03 | 0.5994 ± 0.0336 | 8233.06 ± 0.00      | 72.73 ± 0.52         | 28.40 ± 7.83     |
| full    | LoRA r=50  | 0.85 ± 0.04 | 0.5116 ± 0.2211 | 9005.69 ± 0.00      | 74.04 ± 0.45         | 33.60 ± 6.11     |
| full    | LoRA r=100 | 0.87 ± 0.02 | 0.4565 ± 0.0829 | 10004.11 ± 0.00     | 76.26 ± 1.52         | 33.40 ± 6.19     |
| partial | FFT        | 0.85 ± 0.02 | 0.5172 ± 0.0597 | 1031.86 ± 0.00      | 65.99 ± 0.32         | 23.60 ± 7.13     |
| partial | LoRA r=10  | 0.84 ± 0.01 | 0.4899 ± 0.0777 | 1108.20 ± 0.00      | 67.95 ± 0.46         | 39.60 ± 0.89     |
| partial | LoRA r=50  | 0.86 ± 0.01 | 0.4514 ± 0.0729 | 1148.61 ± 0.00      | 67.48 ± 0.61         | 32.40 ± 5.86     |
| partial | LoRA r=100 | 0.86 ± 0.01 | 0.4357 ± 0.0890 | 1200.48 ± 0.00      | 68.76 ± 0.30         | 28.60 ± 4.28     |

### Comments & open issues (needs further investigation)
Overall, **FFT tends to achieve the best accuracy** across both backbones. This is expected because FFT can update the full parameter space, while LoRA restricts adaptation to a **low-rank subspace**. As a result, **LoRA usually underperforms FFT**, especially at low rank.

A consistent trend is that **LoRA performance improves as rank increases (r=10 → 50 → 100)**, which matches the intuition that higher ranks increase the **expressivity** of the adaptation (less constrained updates), partially closing the gap with FFT.

Training dynamics also change with LoRA. For instance, **ViT (full)** often trains **more epochs** than LoRA on average, while **ViT (partial)** tends to show the opposite (LoRA trains **more epochs** than FFT). For **ResNet**, LoRA frequently trains **more epochs** than FFT in both full and partial setups. This suggests LoRA can lead to **different optimization and early-stopping trajectories** (e.g., smoother loss curves or slower convergence), even under the same max-epoch and early-stopping settings.

Two behaviors deserve further investigation:
1) **LoRA occasionally matches or exceeds FFT in the partial ResNet setting (r=50/100).** A plausible explanation is that constraining updates to a low-rank subspace acts as an **implicit regularizer**, limiting harmful weight drift when only late layers are adapted and improving generalization in some settings. Significance testing and repeated runs are needed to confirm robustness.
2) **GPU memory does not always decrease with LoRA (especially for ViT partial).** Despite a large reduction in trainable parameters, peak GPU memory can **increase**, suggesting the bottleneck is not optimizer state but rather **activation/graph/kernels behavior** (e.g., LoRA wrapping altering attention/linear execution paths or increasing saved tensors), and the fact that “peak allocated” is sensitive to transient spikes. Profiling memory across forward/backward steps is needed to pinpoint the cause.

## Future Work

The next stage of the project will explore generative modeling. The goal is to fine-tune a diffusion model capable of generating dinosaur images conditioned on species labels. Such a model could be used to augment the dataset with synthetic samples and to investigate how generative and discriminative components might be combined to further improve classification performance.
