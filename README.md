# JurassiClass: Dinosaur Species Classification with Convolutional Neural Networks

This repository documents a small deep-learning project focused on classifying dinosaur species from images. The work covers dataset construction, automated and manual data cleaning, model design, transfer learning, and hyperparameter search. 

## Dataset

The dataset was built by scraping images of eight dinosaur species: Microceratus, Pachycephalosaurus, Parasaurolophus, Spinosaurus, Stegosaurus, Triceratops, Tyrannosaurus, and Velociraptor. After downloading, the images were filtered using a CLIP-based cosine-similarity scoring procedure, removing the majority of off-topic content. This still left considerable noise, so a manual curation step followed to discard unrealistic drawings, unrelated animals, and other artifacts. Realistic paleoart was generally preserved, while overly stylized or cartoon-like renderings were removed unless they retained plausible anatomy.

Because dinosaur reconstructions vary significantly across sources and historical periods, different images of the same species may reflect inconsistent interpretations (for example, feathered vs. non-feathered Velociraptors). This variability introduces label noise and makes classification more challenging, but also more interesting from a learning perspective.

## Models

Two convolutional neural networks were implemented using PyTorch Lightning: **RexNet**, based on a ResNet-34 backbone, and **EfficentRex**, based on EfficientNetV2-S. The preprocessing pipeline and training utilities rely on common helper functions such as `letterbox_to_square` and visualization functions from `utils.py`. Cross-validation and grid-search experiments use the training frameworks defined in `CrossValidationTraining.py` :contentReference[oaicite:3]{index=3} and `GridCrossValidation.py` :contentReference[oaicite:4]{index=4}.

Despite having a similar number of total parameters (about 20M), the two models differ in design philosophy: ResNet follows a straightforward residual-block architecture, while EfficientNetV2-S incorporates compound scaling and inverted bottleneck layers, typically achieving better accuracy–efficiency tradeoffs.

## Experiments

The first set of experiments evaluated both models by retraining **only the final classification layer**, keeping the pretrained backbone frozen. Using a learning rate of 0.005 for the classifier, EfficientNetV2-S achieved a mean test accuracy of **0.8247** (std 0.0102) across five folds, while ResNet-34 reached **0.7278** (std 0.0078). Even without fine-tuning, EfficientNet demonstrated a clear advantage, consistent with its stronger feature representations.

Next, both models were fine-tuned on their highest layers in addition to re-training the classifier. The intuition was that ImageNet-pretrained high-level features may not transfer optimally to dinosaur species, so adapting those last layers should improve performance. Both networks were fine-tuned so that the number of trainable parameters remained comparable. With a classifier learning rate of 0.005 and a backbone learning rate of 0.0005, EfficientNet reached **0.9165** accuracy (std 0.007), while ResNet reached **0.8525** (std 0.0250). Fine-tuning significantly improved performance for both, and again EfficientNet remained superior and more stable.

A third experiment added weight decay (0.001) to the AdamW optimizer while keeping learning rates unchanged. This further improved the EfficientNet model to **0.9228** accuracy (std 0.0086). ResNet, however, showed a slight decline to **0.8418** (std 0.0331). At this stage, EfficientNet had a consistent advantage in accuracy, robustness, and sensitivity to regularization. Subsequent experiments therefore focused exclusively on the EfficientNet variant.

## Hyperparameter Search

To refine the training setup, a nested cross-validation procedure was run using two outer folds and two inner folds. This allowed a more reliable evaluation of different learning-rate schedules and weight-decay combinations while limiting GPU cost.

The first search focused on the learning rate for the classifier (“firstlr”) and a scaling factor applied to the fine-tuned backbone layers. Four combinations were tested: classifier learning rates of 0.003 or 0.005, with backbone learning rates reduced by factors of 5 or 10. Among these, the configuration **firstlr = 0.003** with **factor = 10** achieved the best overall accuracy (mean 0.9141, std 0.0178) and was selected for the following weight-decay search.

The second search explored two weight-decay values for the classifier layer and two for the fine-tuned layers. The best configuration was **wd1 = 1e-4** for the classifier and **wd2 = 1e-4** for the backbone. This setting reached an accuracy of **0.9203** (std 0.0118), making it the strongest model tested so far within the nested cross-validation framework.

Overall, EfficientNetV2-S consistently delivered higher accuracy and lower variance than ResNet-34 across all training regimes. Fine-tuning the last block, introducing moderate weight decay, and using a small learning rate for the classifier with a strong reduction for the backbone each contributed to improved performance. These results suggest that the model is able to adapt its high-level representation to the domain of dinosaur imagery despite substantial stylistic variation and noise in the dataset.

## Future Work

The next stage of the project will involve experimenting with generative modeling. In particular, the plan is to fine-tune a diffusion model to generate dinosaur images conditioned on species labels. This could provide a way to augment the dataset with synthetic examples and to study whether generative and discriminative components can be combined to further improve classification accuracy.

