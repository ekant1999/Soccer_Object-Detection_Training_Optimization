# Training Optimization for 2D Object Detection
## Soccer/Football Player Detection using YOLOv8

https://github.com/ekant1999/Soccer_Object-Detection_Training_Optimization/blob/main/Soccer_Object_Detection_Training_Optimization.ipynb

## 1. Objective

This project explores optimization techniques for 2D object detection by improving the training process of a YOLOv8 model on a soccer/football player detection task. Five experiments are conducted to measure the individual and combined impact of architecture modification, training strategy optimization, and custom dataset integration — all evaluated using COCO-style metrics.

![soccer_2](https://github.com/user-attachments/assets/ac14cade-2c25-4644-8055-ac17a91f27ce)
![soccer_3](https://github.com/user-attachments/assets/aaf9c202-8483-4ad5-9fc4-656b2ee9a1d1)


---

## 2. Dataset

### Base Dataset

**Source:** Football Players Detection — [Roboflow Universe](https://universe.roboflow.com)

**Classes (4):** ball, goalkeeper, player, referee

| Split | Images |
|-------|--------|
| Train | 612 |
| Valid | 38 |
| Test | 13 |
| **Total** | **663** |

The dataset consists of broadcast soccer match images with bounding box annotations in YOLO format. The class distribution is heavily skewed toward the "player" class, with "ball" being the rarest and smallest object.

### Custom Annotated Dataset

**Source:** **161 images** were selected with a balanced class distribution:

| Class | Images Selected |
|-------|----------------|
| ball | 65 |
| goalkeeper | 37 |
| referee | 37 |
| player | 22 |

Images were selected to prioritize cleaner annotations (fewer overlapping classes per image). Class IDs were remapped to match the base dataset (e.g., "soccer-player" → "player", custom referee id 2 → base referee id 3). Custom images were split 80/20 into training and validation sets.

### Merged Dataset

| Split | Images |
|-------|--------|
| Train | 740 |
| Valid | 71 |
| Test | 13 |

Merged training set class distribution: ball (575), goalkeeper (464), player (12,345), referee (1,432).

---

## 3. Experiments

Five experiments were conducted to isolate the effect of each optimization. All experiments used random seed 42 and were evaluated on the same validation set.

| # | Experiment | Model | Dataset | Training Strategy |
|---|---|---|---|---|
| 1 | Baseline | YOLOv8n | Base | Default |
| 2 | Architecture | YOLOv8s | Base | Default |
| 3 | Strategy Optimized | YOLOv8s | Base | Optimized |
| 4 | Full + Custom Data | YOLOv8s | Merged | Optimized |
| 5 | Arch + Custom (no strategy) | YOLOv8s | Merged | Default |

### Experiment 1: Baseline

Reference point using YOLOv8n (nano) with all default Ultralytics settings.

| Setting | Value |
|---------|-------|
| Model | YOLOv8n (3.2M parameters) |
| Epochs | 50 |
| Image size | 640 |
| Batch size | 16 |
| Optimizer | SGD (lr=0.01, momentum=0.937) |
| Augmentation | Default (mosaic, horizontal flip, HSV jitter) |
| Loss weights | box=7.5, cls=0.5, dfl=1.5 |

### Experiment 2: Architecture Modification

Upgraded from YOLOv8n to YOLOv8s while keeping all other settings at defaults.

| Aspect | Baseline | Modified |
|--------|----------|----------|
| Model | YOLOv8n | YOLOv8s |
| Parameters | 3.2M | 11.2M |

**Rationale:** Soccer scenes are visually complex with many overlapping players and extremely small objects (ball). The nano model's limited capacity causes underfitting. The small model provides 3.5x more parameters for richer feature extraction.

### Experiment 3: Training Strategy Optimization

Multiple training-level improvements applied on top of YOLOv8s.

**Augmentation changes:**

| Augmentation | Baseline | Optimized | Rationale |
|-------------|----------|-----------|-----------|
| Mosaic | 1.0 | 1.0 | Effective for multi-object scenes; kept |
| Mixup | 0.0 | 0.15 | Blends two images for better generalization |
| CopyPaste | 0.0 | 0.1 | Helps with rare classes (ball, referee) |
| Rotation | 0° | ±10° | Handles different camera angles |
| HSV Saturation | 0.7 | 0.8 | Handles varying lighting conditions |
| HSV Value | 0.4 | 0.5 | Handles shadows on the field |
| Random Erasing | 0.4 | 0.3 | Simulates partial occlusions |

**Optimizer and training changes:**

| Setting | Baseline | Optimized | Rationale |
|---------|----------|-----------|-----------|
| Optimizer | SGD (lr=0.01) | AdamW (lr=0.001) | More stable convergence |
| LR Schedule | Linear decay | Cosine annealing | Smoother learning rate decay |
| Warmup epochs | 3 | 5 | More stable early training |
| Label smoothing | 0.0 | 0.1 | Prevents overconfident predictions |
| Image size | 640 | 736 | Better small object (ball) detection |
| Batch size | 16 | 12 | Adjusted for larger images |
| Epochs | 50 | 80 (patience=15) | More training with early stopping |
| Cls loss weight | 0.5 | 1.0 | Increased classification emphasis |
| Close mosaic | default | Last 10 epochs | Fine-tuning without mosaic |

### Experiment 4: Full Optimization + Custom Data

All optimizations from Experiment 3, trained on the **merged dataset** (base + 161 custom annotated images).

### Experiment 5: Architecture + Custom Data (No Strategy)

YOLOv8s with **default** training settings on the **merged dataset**. This isolates the effect of adding custom data without training strategy changes.

---

## 4. Results

### Overall COCO-Style Evaluation

| Experiment | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|
| 1. Baseline (YOLOv8n) | 0.7200 | 0.4369 | 0.8725 | 0.6520 |
| 2. Architecture (YOLOv8s) | 0.8000 | 0.5248 | 0.9067 | 0.7389 |
| 3. Strategy Optimized | **0.8280** | **0.5548** | **0.9345** | **0.8035** |
| 4. Full + Custom Data | 0.7872 | 0.5378 | 0.8718 | 0.7655 |
| 5. Arch + Custom (no strategy) | 0.7801 | 0.5286 | 0.8767 | 0.7422 |

### Improvement Over Baseline

| Experiment | mAP50 Improvement |
|---|---|
| 2. Architecture (YOLOv8s) | **+11.1%** |
| 3. Strategy Optimized | **+15.0%** |
| 4. Full + Custom Data | **+9.3%** |
| 5. Arch + Custom (no strategy) | **+8.3%** |

### Per-Class mAP50-95 Breakdown

| Class | Baseline | Arch Mod | Strategy | Full+Custom | Arch+Custom |
|---|---|---|---|---|---|
| ball | 0.0261 | 0.1320 | 0.1518 | 0.1554 | 0.1408 |
| goalkeeper | 0.5359 | 0.6429 | 0.6848 | 0.6285 | 0.6656 |
| player | 0.6866 | 0.7463 | 0.7626 | 0.7492 | 0.7342 |
| referee | 0.4989 | 0.5780 | 0.6200 | 0.6183 | 0.5739 |

---

## 5. Analysis

### Most Impactful Change: Architecture Upgrade (+11.1% mAP50)

Switching from YOLOv8n to YOLOv8s produced the single largest improvement. The deeper backbone and wider feature channels enabled richer representations across all classes. mAP50-95 improved from 0.4369 to 0.5248 (+20.1%), indicating significantly better localization. Ball detection saw a 5x improvement (0.0261 → 0.1320).

### Training Strategy: Strong Additional Gains (+3.9% mAP50)

On top of the architecture change, training strategy optimization added 3.9 percentage points to mAP50. The most notable improvement was in recall (0.7389 → 0.8035), meaning the model found objects it previously missed. Key contributors were the AdamW optimizer with cosine LR schedule, enhanced augmentations (Mixup, CopyPaste), and the increased image size (640 → 736) which helped detect the small ball.

### Custom Data: Helps Baseline, But Introduces Domain Noise

Comparing Experiments 2 and 5 isolates the custom data effect with default training: mAP50 dropped slightly from 0.8000 to 0.7801. Similarly, comparing Experiments 3 and 4 shows the custom data's effect with optimized training: mAP50 dropped from 0.8280 to 0.7872.

However, the custom data **did help ball detection** — the ball class achieved its highest mAP50-95 (0.1554) in Experiment 4, surpassing even the strategy-only Experiment 3 (0.1518). The overall performance drop is attributed to domain mismatch: the custom dataset uses "soccer-player" labels with different annotation styles and camera perspectives compared to the base dataset.

### Key Takeaways

1. **Architecture matters most:** YOLOv8n → YOLOv8s contributed the majority of improvement (+11.1%).
2. **Training strategy provides meaningful gains:** AdamW + cosine LR + enhanced augmentations added +3.9%, particularly in recall.
3. **Best model is Experiment 3:** Strategy optimization on the base dataset alone achieves the highest overall performance (0.828 mAP50).
4. **Custom data helps specific classes:** Ball detection improved with custom data, but overall performance dropped due to domain mismatch between datasets.
5. **Data quality > data quantity:** Annotation consistency and domain alignment matter more than simply adding more images.

---

## 6. Conclusion

The combined optimizations resulted in a **15.0% improvement in mAP50** (0.7200 → 0.8280) and a **27.0% improvement in mAP50-95** (0.4369 → 0.5548) over the baseline. Precision improved from 0.8725 to 0.9345 and recall from 0.6520 to 0.8035.

The architecture upgrade was the most impactful single change, while training strategy optimizations provided meaningful additional gains. The custom data experiments demonstrated that naive dataset merging can hurt overall performance due to domain mismatch, even when it helps specific underrepresented classes like ball detection. This highlights that annotation quality, consistency, and domain alignment are critical considerations when augmenting training data.

---

## 7. How to Reproduce

### Requirements

```
pip install ultralytics roboflow pycocotools pandas matplotlib seaborn albumentations
```

### Run

1. Open `Soccer_Object_Detection_Training_Optimization.ipynb` in Google Colab
2. Set runtime to GPU: Runtime → Change runtime type → T4 GPU
3. Run all cells in order
4. Results are saved to `soccer_results/` directory

### Hardware Used

- **GPU:** NVIDIA RTX PRO 6000 Blackwell Server Edition (97 GB VRAM)
- **Framework:** Ultralytics 8.4.37, PyTorch 2.10.0+cu128
- **Environment:** Google Colab

### Project Structure

```
├── Soccer_Object_Detection_Training_Optimization.ipynb
├── README.md
├── experiment_comparison.csv
├── experiment_comparison.png
├── selected_annotation_dataset/
│   ├── ball/       (65 images + labels)
│   ├── goalkeeper/ (37 images + labels)
│   ├── player/     (22 images + labels)
│   ├── referee/    (37 images + labels)
│   └── manifest.json
├── merged_dataset/
│   ├── train/  (740 images)
│   ├── valid/  (71 images)
│   ├── test/   (13 images)
│   └── data.yaml
└── soccer_results/
    ├── exp1_baseline/weights/best.pt
    ├── exp2_architecture/weights/best.pt
    ├── exp3_strategy/weights/best.pt
    ├── exp4_full_custom/weights/best.pt
    ├── exp5_arch_custom/weights/best.pt
    └── predictions/
```

---

## 8. Tools and Frameworks

| Tool | Purpose |
|------|---------|
| [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) | Object detection framework |
| [Roboflow](https://roboflow.com) | Dataset hosting and annotation |
| PyTorch 2.10 + CUDA 12.8 | Deep learning backend |
| Google Colab | Training environment |
| pycocotools | COCO-style evaluation metrics |

---

## 9. References

- Jocher, G., Chaurasia, A., & Qiu, J. (2023). Ultralytics YOLO. https://github.com/ultralytics/ultralytics
- Lin, T.-Y., et al. (2014). Microsoft COCO: Common Objects in Context. https://cocodataset.org
- Loshchilov, I., & Hutter, F. (2019). Decoupled Weight Decay Regularization (AdamW). ICLR 2019.
- Football Players Detection Dataset, Roboflow Universe. https://universe.roboflow.com
- Smart Football Object Detection Dataset, Roboflow Universe. https://universe.roboflow.com
