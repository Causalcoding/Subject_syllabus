# Computer Vision — Interview Prep Syllabus

Computer Vision (CV) is one of the most heavily interviewed specializations across **Data Scientist**, **Machine Learning Engineer**, and **AI Engineer** roles, though the depth expected differs by role:

- **Data Scientists** are typically probed on CV problem framing, classical image processing intuition, evaluation metrics (IoU, mAP, Dice), experiment design, and translating business problems (defect detection, medical imaging, retail shelf analytics) into CV pipelines.
- **Machine Learning Engineers** are expected to go deep on architectures (CNNs, ResNet, YOLO, U-Net), training dynamics, data augmentation, loss functions, and productionizing models (latency, quantization, ONNX/TensorRT export, serving).
- **AI Engineers** increasingly need to know Vision Transformers, multimodal models (CLIP, vision-language models), self-supervised pretraining, and how to integrate vision models into larger AI systems/agents (e.g., feeding image embeddings into an LLM pipeline, building RAG-over-images).

This syllabus goes from pixel-level fundamentals to modern transformer-based and multimodal vision systems, with math, code, pitfalls, and curated interview questions at every stage.

---

## Table of Contents

1. [Image Fundamentals](#image-fundamentals)
2. [Convolutional Neural Networks for Vision](#convolutional-neural-networks-for-vision)
3. [Object Detection](#object-detection)
4. [Image Segmentation](#image-segmentation)
5. [Vision Transformers and Modern Architectures](#vision-transformers-and-modern-architectures)
6. [Video, Face, OCR, Pose, and 3D Vision](#video-face-ocr-pose-and-3d-vision)
7. [Practical CV](#practical-cv)
8. [Rapid-Fire Interview Q&A](#rapid-fire-interview-qa)

---

## Image Fundamentals

### Digital image representation: pixels, channels, color spaces

**Intuition.** A digital image is a discretized sampling of a continuous light signal projected onto a sensor grid. Each cell in the grid is a **pixel** (picture element), storing intensity value(s). A grayscale image is a 2D matrix `H x W`; a color image is a 3D tensor `H x W x C` where `C` is the number of channels (3 for RGB).

**Bit depth.** Most images use 8 bits/channel (`0–255`), giving `256^3 ≈ 16.7M` colors for RGB. Medical/scientific images often use 16-bit (`0–65535`) for higher dynamic range.

**Color spaces:**

| Color space | Channels | Description | When used |
|---|---|---|---|
| RGB | R, G, B | Additive primary colors; hardware-native (sensors, displays) | Default for most CV models |
| Grayscale | Intensity | Weighted sum: `Y = 0.299R + 0.587G + 0.114B` (luminance) | Edge detection, classical CV, reducing compute |
| HSV | Hue, Saturation, Value | Separates color (hue) from intensity (value); Hue is circular (0–360°) | Color-based segmentation, robust to lighting changes |
| YCbCr | Luma, Chroma-blue, Chroma-red | Separates luminance from chrominance | JPEG/video compression |
| LAB | Lightness, a (green-red), b (blue-yellow) | Perceptually uniform — Euclidean distance ≈ perceived color difference | Color correction, perceptual similarity |

**Why HSV matters practically:** thresholding "red objects" in RGB is unstable under shadows/lighting because R, G, B all shift together with brightness. In HSV, Hue stays roughly constant regardless of lighting; only Value changes. This is why classical color-based segmentation (e.g., traffic light detection, skin detection) is done in HSV.

```python
import cv2
import numpy as np

img_bgr = cv2.imread("image.jpg")           # OpenCV loads as BGR, not RGB!
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

# HSV-based color thresholding (e.g., isolate "red" objects)
lower_red = np.array([0, 120, 70])
upper_red = np.array([10, 255, 255])
mask = cv2.inRange(img_hsv, lower_red, upper_red)
```

**Pitfalls:**
- OpenCV reads images in **BGR**, not RGB — a classic bug source when mixing OpenCV with PIL/matplotlib/PyTorch (which expect RGB).
- Normalizing pixel values: always confirm whether a pretrained model expects `[0,1]`, `[-1,1]`, or ImageNet-mean/std normalization — mismatches silently degrade accuracy without throwing errors.
- Grayscale conversion loses color information irreversibly; don't apply it upstream of a task that depends on color cues.

**Practical tips:**
- Always inspect `img.shape`, `img.dtype` before feeding into a pipeline — silent `uint8` vs `float32` mismatches are common.
- For perceptual color similarity/clustering, prefer LAB or a Delta-E metric over raw RGB Euclidean distance.

---

### Image processing basics: filtering/convolution, edge detection, histogram equalization, morphological operations

**Convolution / filtering.** A filter (kernel) is a small matrix slid over the image; at each position, an element-wise multiply-and-sum produces the output pixel. This is the mathematical operation:

```
(I * K)(x, y) = Σ_i Σ_j I(x+i, y+j) · K(i, j)
```

Technically image processing uses **cross-correlation** (no kernel flip), while true mathematical "convolution" flips the kernel; in ML frameworks the term "convolution" is used loosely for both.

**Common kernels:**

| Kernel | Purpose | Example (3x3) |
|---|---|---|
| Box blur | Smoothing/noise reduction | all `1/9` |
| Gaussian blur | Smoothing, weighted by distance from center | Gaussian-weighted |
| Sobel (x, y) | Gradient / edge detection | `[[-1,0,1],[-2,0,2],[-1,0,1]]` |
| Laplacian | Second derivative, edge/blob detection | `[[0,1,0],[1,-4,1],[0,1,0]]` |
| Sharpen | Enhance edges | `[[0,-1,0],[-1,5,-1],[0,-1,0]]` |

**Sobel edge detection:** computes approximate image gradients `Gx, Gy` via convolution with horizontal/vertical kernels. Gradient magnitude `G = sqrt(Gx² + Gy²)` highlights edges; gradient direction `θ = atan2(Gy, Gx)` gives edge orientation.

**Canny edge detector** — the classical gold-standard, a multi-stage pipeline:
1. **Gaussian smoothing** to reduce noise.
2. **Gradient computation** (Sobel) → magnitude and direction.
3. **Non-maximum suppression**: thin edges by keeping only local maxima along the gradient direction.
4. **Double thresholding**: pixels above `high_thresh` are "strong" edges; below `low_thresh` are discarded; in-between are "weak."
5. **Edge tracking by hysteresis**: weak edges are kept only if connected to a strong edge.

```python
import cv2

gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, threshold1=100, threshold2=200)
```

**Histogram equalization.** Redistributes pixel intensities to flatten (approximately uniformize) the histogram, improving contrast in poorly lit/low-contrast images. Computed via the cumulative distribution function (CDF) of pixel intensities:

```
CDF(k) = Σ_{i=0}^{k} h(i) / N
new_pixel = round(CDF(old_pixel) * (L-1))
```

where `h` is the histogram, `N` total pixels, `L` = number of intensity levels (256).

```python
equalized = cv2.equalizeHist(gray)

# Adaptive version (CLAHE) — handles local contrast variation better than global equalization
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
equalized_adaptive = clahe.apply(gray)
```

**Morphological operations** — operate on binary (or grayscale) images using a structuring element:

| Operation | Effect | Formula intuition |
|---|---|---|
| Erosion | Shrinks white regions, removes small noise | Pixel stays "on" only if entire kernel window fits inside foreground |
| Dilation | Grows white regions, fills small holes | Pixel turns "on" if any kernel window pixel touches foreground |
| Opening | Erosion → Dilation | Removes small noise while preserving overall shape |
| Closing | Dilation → Erosion | Fills small holes/gaps while preserving overall shape |

```python
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
eroded = cv2.erode(binary_img, kernel, iterations=1)
dilated = cv2.dilate(binary_img, kernel, iterations=1)
opened = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
closed = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, kernel)
```

**Pitfalls:**
- Applying Canny/Sobel directly on noisy images without smoothing produces spurious edges.
- Global histogram equalization can over-amplify noise in flat regions and wash out already-good-contrast images — CLAHE is usually safer.
- Kernel size/threshold choices are highly image-dependent; hardcoded thresholds rarely generalize across datasets/lighting conditions.

**Practical tips:**
- In interviews, be ready to derive why Sobel approximates a derivative (finite difference approximation of `∂I/∂x`).
- Know that these classical operations are still used heavily as **preprocessing** steps even in deep-learning pipelines (e.g., document scanning, medical image preprocessing, OCR).

---

### Feature descriptors (classical): SIFT, HOG, ORB

Before deep learning, computer vision relied on **hand-engineered feature descriptors** to convert local image patches into fixed-length vectors that are robust to viewpoint, scale, illumination changes — used for matching, retrieval, and classification with classical ML (SVMs, etc.).

**SIFT (Scale-Invariant Feature Transform):**
- Detects **keypoints** at multiple scales using a **Difference-of-Gaussians (DoG)** pyramid (approximates Laplacian of Gaussian, which is expensive to compute directly).
- For each keypoint, computes a dominant orientation (rotation invariance) and builds a 128-dimensional descriptor from local gradient histograms (4x4 grid × 8 orientation bins).
- **Invariances:** scale, rotation, partially illumination and affine transforms.
- Historically patented; slower than modern alternatives, but still a gold-standard for its robustness (used in panorama stitching, 3D reconstruction/SfM, image matching).

**HOG (Histogram of Oriented Gradients):**
- Divides the image into small cells, computes gradient orientation histograms per cell, then normalizes over larger blocks (to handle illumination variation).
- Captures **shape/edge structure** rather than keypoints — famously used with a linear SVM for pedestrian/person detection (Dalal & Triggs, 2005), a precursor to modern object detection.

```python
from skimage.feature import hog
features, hog_image = hog(gray_img, orientations=9, pixels_per_cell=(8, 8),
                           cells_per_block=(2, 2), visualize=True)
```

**ORB (Oriented FAST and Rotated BRIEF):**
- Combines the **FAST** keypoint detector (fast corner detection via intensity comparison in a circular pattern) with **BRIEF** descriptors (binary strings from pixel-intensity comparisons), plus orientation for rotation invariance.
- Free/open-source alternative to SIFT/SURF, much faster (binary descriptors → Hamming distance matching), slightly less robust to large scale/viewpoint changes.
- Common in real-time applications: SLAM, AR tracking, mobile vision.

```python
import cv2
orb = cv2.ORB_create()
keypoints, descriptors = orb.detectAndCompute(gray, None)

# Matching between two images
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(descriptors1, descriptors2)
matches = sorted(matches, key=lambda m: m.distance)
```

**Why deep learning replaced these for most tasks:**
- Hand-crafted descriptors encode a fixed, human-designed notion of "useful structure" (edges/gradients/corners). CNNs **learn** hierarchical, task-specific features directly from data, often discovering representations that outperform hand-crafted ones on semantic tasks (classification, detection, segmentation).
- Classical descriptors are strong for **geometric matching** tasks (structure-from-motion, panorama stitching, SLAM, visual odometry) where precise, interpretable, low-level correspondence matters and training data is limited — deep learning has not fully displaced them here.
- Deep features generalize across appearance variation far better (viewpoint, occlusion, intra-class variation) because they are trained end-to-end on the target objective, whereas SIFT/HOG/ORB are generic and not optimized for any specific downstream semantic task.
- Deep learned local descriptors (SuperPoint, D2-Net) now compete with/replace SIFT even in geometric matching pipelines.

**Pitfalls:**
- Assuming classical descriptors are "obsolete" everywhere — they remain relevant in low-data regimes, real-time embedded systems, and geometric vision (SLAM/AR) where CNNs are overkill or lack precise correspondence.
- Not normalizing HOG blocks — illumination sensitivity reappears without block normalization.

**Practical tips:** Interviewers often ask you to compare/contrast SIFT vs. learned features conceptually — focus on invariances, computational cost, and data requirements rather than implementation minutiae.

### Interview Questions

1. **Q: What is the difference between a pixel and a channel?**
   A: A pixel is a single spatial location in an image grid; a channel is one "layer" of information at every pixel (e.g., Red, Green, Blue). An RGB image of size `H x W` has `H*W` pixels, each described by 3 channel values, giving a tensor of shape `H x W x 3`.

2. **Q: Why does OpenCV load images in BGR instead of RGB, and why does it matter?**
   A: Historical legacy from early Windows camera/codec APIs that stored bytes in BGR order. It matters because feeding a BGR image into a model or visualization tool expecting RGB silently swaps the red and blue channels, producing wrong colors and potentially degrading model accuracy without an explicit error — always convert with `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)`.

3. **Q: Explain how the Sobel operator detects edges.**
   A: Sobel convolves the image with two 3x3 kernels approximating the partial derivatives in x and y directions. The gradient magnitude `sqrt(Gx²+Gy²)` is large where intensity changes sharply (edges), and gradient direction indicates edge orientation. It's essentially a discrete, noise-robust finite-difference approximation of the image gradient.

4. **Q: Walk through the Canny edge detection algorithm.**
   A: (1) Gaussian blur to suppress noise, (2) compute gradients via Sobel to get magnitude and direction, (3) non-maximum suppression to thin edges to single-pixel width along the gradient direction, (4) double thresholding to classify pixels as strong/weak/non-edges, (5) hysteresis edge tracking to keep weak edges only if connected to strong ones. This produces clean, thin, well-localized edges.

5. **Q: What problem does histogram equalization solve, and what's a limitation?**
   A: It improves global contrast in images with a narrow/skewed intensity distribution by remapping intensities so the histogram is more uniformly spread (using the CDF as the mapping function). Limitation: it's global, so it can over-enhance noise in already-good-contrast regions and doesn't adapt to local contrast variation — CLAHE (adaptive, tile-based) addresses this.

6. **Q: Explain erosion vs. dilation and give a practical use case for each.**
   A: Erosion shrinks foreground regions (keeps a pixel on only if the whole structuring element fits within the foreground) — used to remove small noise specks. Dilation grows foreground regions (turns a pixel on if any part of the structuring element touches the foreground) — used to fill small holes/gaps or connect nearby components. Opening (erode→dilate) removes noise while preserving shape; closing (dilate→erode) fills holes while preserving shape.

7. **Q: What makes SIFT scale-invariant?**
   A: SIFT builds a scale-space pyramid using Difference-of-Gaussians (DoG) at multiple octaves/scales, detecting keypoints that are extrema across both spatial location and scale. A keypoint detected robustly across multiple scales is assigned the scale at which it was most stable, making the descriptor invariant to the image's overall zoom level.

8. **Q: How does HOG differ conceptually from SIFT?**
   A: SIFT is a sparse, keypoint-based descriptor designed for matching/correspondence at distinctive local points. HOG is a dense, block-based descriptor computed over a fixed grid across the whole image (or detection window), capturing overall shape/silhouette via gradient orientation histograms — better suited to whole-object detection (e.g., pedestrians) than point matching.

9. **Q: Why is ORB preferred over SIFT in real-time applications like SLAM?**
   A: ORB uses FAST for fast corner detection and BRIEF binary descriptors, which are compact (256 bits) and can be compared via Hamming distance (XOR + popcount) — orders of magnitude faster than SIFT's floating-point Euclidean distance matching on 128-d vectors. This speed tradeoff sacrifices some robustness to large scale/affine changes but is well worth it for real-time frame-rate constraints.

10. **Q: In what scenarios would you still choose classical CV features over a deep learning approach today?**
    A: When: (a) training data is extremely limited or unavailable, (b) real-time performance on constrained embedded hardware is required with no GPU, (c) the task is precise geometric matching/correspondence (SLAM, structure-from-motion, camera calibration) where sub-pixel accuracy and interpretability matter, (d) the domain is not well represented in pretrained model distributions and fine-tuning isn't feasible, or (e) explainability/auditability is a hard requirement.

11. **Q: What is non-maximum suppression in the context of Canny edge detection (not object detection)?**
    A: After computing the gradient magnitude and direction at every pixel, NMS keeps a pixel only if its magnitude is a local maximum compared to its two neighbors along the gradient direction; otherwise it's suppressed to 0. This thins wide gradient ridges into single-pixel-wide edges.

12. **Q: Derive/explain why grayscale conversion uses weights approximately `0.299, 0.587, 0.114` for R, G, B.**
    A: These weights (ITU-R BT.601 standard) approximate the human eye's differing sensitivity to each color — the eye is most sensitive to green, less to red, least to blue — so luminance is a weighted sum reflecting perceived brightness rather than a simple average.

13. **Q: What's the difference between erosion/dilation (morphology) and Gaussian blur (filtering)?**
    A: Both use a sliding window, but morphological ops operate on binary/thresholded images with set-theoretic min/max logic over a structuring element (shape-preserving, non-linear), while Gaussian blur is a linear, weighted-average convolution applied to continuous intensity values for smoothing/noise reduction — not designed to preserve shape boundaries.

14. **Q: How would you detect a rotated version of a template image using classical CV?**
    A: Use a rotation-invariant descriptor pipeline: detect keypoints with SIFT/ORB (which assign a dominant orientation to each keypoint and build a rotation-normalized descriptor), then match descriptors between the template and target image using nearest-neighbor matching (with a distance ratio test, e.g., Lowe's ratio test) followed by RANSAC to estimate a robust homography.

15. **Q: What is the Laplacian operator and how does it differ from Sobel for edge detection?**
    A: The Laplacian is a second-derivative operator (`∇²I = ∂²I/∂x² + ∂²I/∂y²`) that is isotropic (rotation-invariant, no directional bias) and detects edges as zero-crossings, but is more noise-sensitive since second derivatives amplify high-frequency noise. Sobel is a first-derivative, directional operator, generally more robust for practical edge detection; Laplacian is often combined with Gaussian smoothing first (Laplacian of Gaussian, LoG) to mitigate noise sensitivity.

---

## Convolutional Neural Networks for Vision

### Convolution/pooling recap in vision context, receptive field growth

**Convolutional layer.** A learnable filter (kernel), e.g., `3x3xC_in`, slides across the input feature map producing one output channel per filter via a dot product at each spatial location. Key properties that make CNNs suited to images:

- **Local connectivity** — each output depends only on a local neighborhood (unlike fully-connected layers).
- **Parameter sharing** — the same filter weights are reused across all spatial positions, drastically reducing parameters vs. a dense layer and encoding **translation equivariance** (a shifted input produces a correspondingly shifted output).
- **Hierarchical feature composition** — stacking layers lets the network build from edges → textures → parts → objects.

**Output size formula:**
```
Output = floor((W - K + 2P) / S) + 1
```
where `W` = input size, `K` = kernel size, `P` = padding, `S` = stride.

```python
import torch.nn as nn
conv = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
# padding=1 with kernel=3, stride=1 => "same" spatial size (common design pattern)
```

**Pooling.** Downsamples feature maps, reducing spatial resolution while retaining salient information and adding a degree of translation invariance.

| Type | Operation | Effect |
|---|---|---|
| Max pooling | Take max in window | Preserves strongest activation, common default |
| Average pooling | Take mean in window | Smoother, less commonly used mid-network |
| Global Average Pooling (GAP) | Average entire feature map to 1 value/channel | Used before final classification layer (ResNet, etc.) instead of flatten+FC |

**Receptive field (RF).** The region of the *original input image* that influences a given output unit. RF grows as you stack layers:

```
RF_l = RF_{l-1} + (K_l - 1) * stride_product_{<l}
```

Intuition: a single 3x3 conv sees a 3x3 patch. Stack two 3x3 convs (stride 1) and the second layer's output "sees" a 5x5 region of the original input, because each of its 3x3 inputs already saw a 3x3 region, and these overlap and compose. This is why deep networks with small kernels (3x3, VGG-style) can achieve very large effective receptive fields with fewer parameters than a single large-kernel layer, while also gaining more non-linearities (more ReLUs = more expressive power) for the same number of parameters.

Pooling/stride also expands RF faster (a stride-2 layer doubles the RF growth rate of subsequent layers), which is why deep CNNs interleave stride/pooling — to reach receptive fields covering the whole image by the final layers, necessary for tasks needing global context (classification).

**Pitfalls:**
- Confusing receptive field with "actual output resolution" — a large RF doesn't mean uniform/dense information capture; effective RF (empirically, contribution-weighted) is usually much smaller than the theoretical maximum, concentrated near the center (Gaussian-like falloff).
- Forgetting that 1x1 convolutions don't grow spatial receptive field at all — they mix channel information only (used for dimensionality reduction, e.g., in Inception/ResNet bottlenecks).

**Practical tips:** When asked to compute RF growth, always account for stride multiplicatively across the network, not additively.

---

### Key architectures and their innovations

| Architecture | Year | Key Innovation | Why it mattered |
|---|---|---|---|
| LeNet-5 | 1998 | First practical CNN (conv+pool+FC) for digit recognition | Proved CNNs work for vision (MNIST) |
| AlexNet | 2012 | ReLU, dropout, GPU training, large-scale ImageNet training | Kicked off the deep learning revolution (ImageNet win by huge margin) |
| VGG | 2014 | Very deep (16–19 layers), uniform 3x3 convs stacked | Showed depth + small filters > shallow + large filters |
| ResNet | 2015 | Residual/skip connections | Solved vanishing gradient in very deep nets; enabled 100+ layer training |
| Inception (GoogLeNet) | 2014 | Multi-scale parallel convs (1x1, 3x3, 5x5) in one block | Efficient multi-scale feature extraction, fewer params via 1x1 bottlenecks |
| DenseNet | 2017 | Dense connections — each layer receives all previous layers' outputs | Strong gradient flow, feature reuse, fewer parameters for given accuracy |
| EfficientNet | 2019 | Compound scaling (depth, width, resolution jointly) via NAS | Best accuracy/FLOPs tradeoff at the time |
| MobileNet | 2017 | Depthwise separable convolutions | Massive FLOPs/parameter reduction for mobile/edge deployment |

**LeNet-5.** Simple conv → pool → conv → pool → FC → FC stack for handwritten digit recognition (MNIST). Established the basic CNN template still used today.

**AlexNet.** 8 layers, used ReLU (faster convergence than sigmoid/tanh, mitigates vanishing gradients), dropout for regularization, data augmentation, and trained on 2 GPUs. Won ILSVRC 2012 by a large margin over classical methods, proving deep CNNs' superiority at scale.

**VGG.** Used only 3x3 convolutions stacked deeply (16/19 layers), showing that depth with small, uniform filters generalizes better than fewer large-filter layers, at the cost of many parameters (~138M for VGG16) — most of which sit in the fully-connected layers.

**ResNet — residual learning.** Core idea: instead of learning `H(x)` directly, learn a residual `F(x) = H(x) - x`, so the block computes:
```
y = F(x, W) + x
```
This **skip/identity connection** lets gradients flow directly through the shortcut during backpropagation (`∂y/∂x` includes the identity term, so gradients don't vanish even if `∂F/∂x` is small), enabling stable training of very deep networks (50, 101, 152, even 1000+ layers). Intuitively, if extra layers aren't helpful, the network can learn `F(x) ≈ 0` and just pass the identity through — so adding depth can't hurt (in theory), which also solves the empirically observed "degradation problem" where very deep non-residual nets performed *worse* than shallower ones.

```python
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + identity          # the residual/skip connection
        return self.relu(out)
```

**Inception — multi-scale.** An Inception block applies 1x1, 3x3, 5x5 convolutions (and pooling) **in parallel** on the same input and concatenates the outputs, letting the network choose the most useful scale of features at each layer, rather than committing to one filter size per layer. 1x1 convolutions are used as "bottlenecks" before the expensive 3x3/5x5 convs to reduce channel depth first, cutting computation substantially.

**DenseNet — dense connections.** Each layer receives the concatenated feature maps of *all preceding layers* in its "dense block" (not just the immediately prior one, unlike ResNet's additive skip), and passes its own output forward to all subsequent layers. This maximizes feature reuse and gradient flow, often achieving ResNet-level accuracy with fewer parameters, though it uses more memory (feature map concatenation growing per layer).

**EfficientNet — compound scaling.** Prior work scaled depth, width, or resolution independently (ad hoc). EfficientNet's insight: scale all three dimensions **jointly** with a fixed ratio found via small-scale grid search then applied through a compound coefficient `φ`:
```
depth: d = α^φ,  width: w = β^φ,  resolution: r = γ^φ,   subject to α·β²·γ² ≈ 2
```
Combined with a Neural-Architecture-Search-derived base network (EfficientNet-B0, built from MBConv/depthwise-separable blocks with squeeze-and-excitation), this gave a family (B0–B7) with state-of-the-art accuracy per FLOP.

**MobileNet — depthwise separable convolutions.** Splits a standard convolution into two cheaper steps:
1. **Depthwise conv**: apply one `k x k` filter *per input channel independently* (no cross-channel mixing).
2. **Pointwise conv**: a `1x1` conv across channels to combine information.

Standard conv cost: `H·W·C_in·C_out·K²`. Depthwise-separable cost: `H·W·C_in·K² + H·W·C_in·C_out` — roughly `1/C_out + 1/K²` of the original cost, e.g., for `K=3, C_out=256` this is roughly an 8–9x reduction in FLOPs/parameters, crucial for mobile/edge inference.

```python
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, stride,
                                    padding=kernel_size // 2, groups=in_ch)
        self.pointwise = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))
```

**Pitfalls:**
- Thinking ResNet's skip connection changes the *output dimensionality* — when input/output channel counts differ (e.g., at stage transitions), a 1x1 conv "projection shortcut" is needed to match dimensions before adding.
- Assuming deeper is always better — without residual connections or proper normalization, very deep plain CNNs suffer vanishing/exploding gradients and degrade in accuracy.
- Forgetting that depthwise separable convolutions trade a small accuracy drop for large efficiency gains — not a free lunch, just a good tradeoff for constrained deployment.

**Practical tips:** Be ready to sketch the ResNet residual equation and explain the gradient flow argument on a whiteboard — this is one of the most common CV architecture interview questions.

---

### Transfer learning in CV: feature extraction vs fine-tuning, choosing which layers to freeze

**Why transfer learning works in vision.** Early CNN layers learn generic, broadly reusable features (edges, colors, textures, simple shapes) largely independent of the final task; later layers learn increasingly task/dataset-specific, semantic features (object parts, class-discriminative patterns). Pretraining on a large diverse dataset (ImageNet, etc.) gives a strong generic feature extractor that transfers well to new but related tasks, especially when the target dataset is small.

**Two main strategies:**

| Strategy | What happens | When to use |
|---|---|---|
| **Feature extraction** | Freeze the entire pretrained backbone; only train a new head (classifier) on top | Small target dataset, target domain similar to pretraining domain, limited compute |
| **Fine-tuning** | Unfreeze some/all backbone layers and continue training (usually with a small learning rate) | Larger target dataset, target domain differs somewhat from pretraining domain, more compute available |

**Choosing which layers to freeze — rules of thumb:**
- **Freeze early layers** (generic features: edges/colors/textures) — these transfer well regardless of task and rarely need updating.
- **Unfreeze/fine-tune later layers** (semantic, task-specific features) — these need adaptation to the new class distribution/domain.
- The more your target domain differs from the pretraining domain (e.g., ImageNet natural images → X-ray images), the *more* layers you should unfreeze, even potentially the whole network, but at a low learning rate to avoid catastrophic forgetting of useful pretrained weights.
- The smaller your target dataset, the *fewer* layers you should unfreeze (to avoid overfitting the large capacity of unfrozen layers to too little data).

```python
import torchvision.models as models
import torch.nn as nn

model = models.resnet50(weights="IMAGENET1K_V2")

# Strategy 1: feature extraction — freeze everything except the new head
for param in model.parameters():
    param.requires_grad = False
model.fc = nn.Linear(model.fc.in_features, num_classes)  # new head is trainable by default

# Strategy 2: fine-tune only the last residual stage + head
for name, param in model.named_parameters():
    param.requires_grad = "layer4" in name or "fc" in name

# Strategy 3: full fine-tune with a low LR, often using discriminative LR
# (lower LR for early layers, higher for later layers/head)
optimizer = torch.optim.Adam([
    {"params": model.layer1.parameters(), "lr": 1e-5},
    {"params": model.layer2.parameters(), "lr": 1e-5},
    {"params": model.layer3.parameters(), "lr": 5e-5},
    {"params": model.layer4.parameters(), "lr": 1e-4},
    {"params": model.fc.parameters(), "lr": 1e-3},
])
```

**Practical workflow:** 
1. Start with feature extraction (fast, cheap, good baseline).
2. If underfitting / accuracy plateau, progressively unfreeze from the top (later layers) down, monitoring validation performance ("progressive unfreezing," popularized by ULMFiT/fastai).
3. Always use a much smaller learning rate for fine-tuning than for training from scratch — pretrained weights are already near a good optimum and large updates can destroy them.
4. Re-compute/adjust `BatchNorm` running statistics carefully — when freezing, keep BN layers in eval mode (using pretrained running stats) unless fine-tuning with enough data to re-estimate them; small-batch fine-tuning with BN in train mode can destabilize training.

**Pitfalls:**
- Using the same learning rate for pretrained backbone and new head — the head starts randomly initialized and needs faster/larger updates than the already-trained backbone.
- Forgetting to match preprocessing (input normalization, image size) to what the backbone was pretrained with — a mismatch silently degrades transfer performance.
- Fine-tuning the entire network on a tiny dataset — leads to overfitting/catastrophic forgetting of general features.
- Leaving `BatchNorm` layers in training mode with very small batch sizes during fine-tuning, causing unstable statistics.

### Interview Questions

1. **Q: Explain the intuition behind why CNNs work well for images compared to fully-connected networks.**
   A: CNNs exploit two image-specific priors: locality (nearby pixels are more related than distant ones, so local receptive fields suffice) and translation equivariance (an object's identity doesn't depend on its position, so sharing filter weights across spatial locations is valid and drastically reduces parameters vs. a dense layer connecting every pixel to every neuron).

2. **Q: What is the receptive field of a neuron, and how does stacking layers grow it?**
   A: The receptive field is the region of the original input image that can influence a given neuron's activation. Stacking convolutional layers grows the RF because each layer's output depends on a local window of the *previous* layer's output, which itself already aggregated information from a local window of the layer before it — RF compounds multiplicatively with stride and additively with kernel size across depth.

3. **Q: Why did ResNet's residual connections solve the "degradation problem" in very deep networks?**
   A: Without shortcuts, gradients must flow through many stacked non-linear transformations, and can vanish (or explode) as depth increases, making optimization difficult — deeper plain networks were empirically *worse* than shallower ones, not due to overfitting but due to optimization difficulty. The identity shortcut `y = F(x) + x` provides a direct path for gradients (`∂y/∂x` includes an identity term), so even if `F` learns something unhelpful, the network can default to passing `x` through unchanged, ensuring depth is never worse than a shallower equivalent in principle.

4. **Q: Compare VGG and Inception architecturally. What tradeoff does each represent?**
   A: VGG stacks uniform 3x3 convolutions very deeply — simple, effective, but parameter-heavy (mostly in FC layers) and computationally expensive. Inception uses parallel multi-scale convolutions (1x1, 3x3, 5x5) within a block plus 1x1 bottlenecks to reduce channels before expensive convolutions, achieving similar or better accuracy with far fewer FLOPs/parameters by being more computationally efficient per unit of representational power.

5. **Q: What is a depthwise separable convolution, and why is it more efficient than a standard convolution?**
   A: It factorizes a standard convolution into a depthwise step (a separate spatial filter per input channel, no cross-channel mixing) followed by a pointwise 1x1 convolution (mixes channels). This reduces computation from `H·W·C_in·C_out·K²` to roughly `H·W·C_in·K² + H·W·C_in·C_out`, cutting FLOPs by roughly `1/C_out + 1/K²`, which is why MobileNet uses it for efficient mobile inference.

6. **Q: What is compound scaling in EfficientNet, and why is it better than scaling only depth or only width?**
   A: Compound scaling jointly increases network depth, width, and input resolution using a single compound coefficient `φ` with fixed ratios (`α, β, γ`) found empirically, rather than scaling one dimension in isolation. Scaling only depth risks vanishing gradients and diminishing returns; scaling only width fails to capture higher-level features well; scaling only resolution without more capacity to process it wastes the extra detail. Balanced joint scaling captures accuracy gains more efficiently per added FLOP.

7. **Q: When would you choose feature extraction over fine-tuning for a transfer learning problem?**
   A: When the target dataset is small (high overfitting risk if unfreezing large capacity), the target domain is visually similar to the pretraining domain (e.g., another natural-image classification task), or compute/time budget is limited — feature extraction (frozen backbone + trainable head) gives a strong, fast, low-risk baseline.

8. **Q: You're fine-tuning a pretrained ImageNet model on a medical X-ray dataset with only 2,000 images. What specific choices would you make?**
   A: Use a smaller, more regularized backbone or freeze most early/middle layers and only fine-tune the last block(s)/head to limit trainable capacity relative to data size; use a small learning rate and possibly differential learning rates (lower for backbone, higher for head); apply strong but domain-appropriate augmentation (careful with flips — laterality can matter medically); use early stopping and cross-validation; consider that ImageNet features (natural images) may transfer less well to X-rays (grayscale, very different statistics) than to another natural-image dataset, so a domain-relevant pretrained model (e.g., pretrained on medical images, if available) may transfer better.

9. **Q: Why is a 1x1 convolution useful, given it doesn't enlarge the receptive field?**
   A: A 1x1 convolution acts purely across the channel dimension at each spatial location — it can reduce or expand the number of channels (as a learned linear projection + nonlinearity), enabling cheap dimensionality reduction/bottlenecking (as in Inception and ResNet bottleneck blocks) before expensive spatial convolutions, without touching spatial resolution or receptive field.

10. **Q: What happens to BatchNorm layers when you freeze a backbone for transfer learning, and why does it matter?**
    A: If you freeze the backbone but leave BatchNorm layers in training mode, they'll keep updating running mean/variance statistics using potentially small/unrepresentative batches from your new dataset, which can destabilize the frozen feature representations. Best practice is to set the frozen backbone (including its BN layers) to `eval()` mode so BN uses the pretrained running statistics, unless you have enough data/batch size to justify re-estimating them.

11. **Q: What is "progressive unfreezing" and why might it outperform unfreezing everything at once?**
    A: Progressive unfreezing gradually unfreezes layers from the top (task-specific, later layers) down toward the bottom (generic, early layers) over successive training phases, rather than unfreezing the whole network immediately. This avoids destructively large gradient updates propagating into well-tuned generic early-layer features before the new head has stabilized, generally leading to more stable convergence and better final performance, especially on smaller datasets.

12. **Q: Explain how DenseNet differs from ResNet in how it uses skip connections.**
    A: ResNet uses additive identity shortcuts — each block outputs `F(x) + x`, combining information via summation, keeping the number of channels the same across the shortcut. DenseNet uses concatenation — each layer's output is concatenated with (not added to) all previous layers' outputs within a dense block, so channel count grows with depth, maximizing feature reuse and gradient flow (every layer has a direct path to the loss) at the cost of higher memory usage.

13. **Q: Given two GPUs and needing to deploy a vision model on a mobile device, would you pick VGG16 or MobileNetV2? Justify.**
    A: MobileNetV2 — it uses depthwise separable convolutions (and inverted residuals) giving comparable or better accuracy-per-FLOP than VGG16 while using roughly 30-50x fewer parameters and FLOPs, which is essential for latency and battery constraints on mobile hardware; VGG16's dense fully-connected layers and heavy standard convolutions are far too costly for edge deployment.

14. **Q: What's the difference between Global Average Pooling and Flatten before a classification head, and why do modern architectures (ResNet, etc.) prefer GAP?**
    A: Flatten preserves all spatial positions as separate features feeding into a large fully-connected layer (many parameters, position-sensitive, prone to overfitting, requires fixed input size). GAP averages each channel's entire feature map into a single value, producing a compact, position-invariant, fixed-size vector regardless of input resolution, with zero additional parameters — this reduces overfitting risk and allows the network to handle variable input sizes.

15. **Q: If your fine-tuned model's training loss decreases but validation accuracy is worse than the frozen-backbone baseline, what would you investigate?**
    A: Likely overfitting from unfreezing too much capacity relative to dataset size — check if the learning rate for the backbone is too high (destroying pretrained weights, aka catastrophic forgetting), verify preprocessing/normalization matches pretraining, check for data leakage or too little augmentation, and consider re-freezing more layers or adding stronger regularization (weight decay, dropout, augmentation) or reverting to a partial-freeze / discriminative learning rate strategy.

---

## Object Detection

### Problem formulation: bounding boxes, IoU, non-max suppression

**Object detection** = localize (bounding box) + classify every instance of interest in an image. Output per detected object: `(x_min, y_min, x_max, y_max, class, confidence)` or center-format `(x_center, y_center, width, height, class, confidence)`.

**Intersection over Union (IoU)** — the standard metric for how well a predicted box matches a ground-truth box:

```
IoU = Area(Box_pred ∩ Box_gt) / Area(Box_pred ∪ Box_gt)
```

```python
def iou(boxA, boxB):
    # box format: [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_area = max(0, xB - xA) * max(0, yB - yA)
    boxA_area = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxB_area = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return inter_area / float(boxA_area + boxB_area - inter_area + 1e-9)
```

An IoU of 1.0 = perfect overlap; 0 = no overlap. A prediction is typically counted as a "true positive" if `IoU >= threshold` (commonly 0.5) **and** the class matches.

**Non-Max Suppression (NMS).** Detectors typically emit many overlapping candidate boxes for the same object (especially with dense anchors/grid cells). NMS removes redundant boxes:

1. Sort all candidate boxes by confidence score, descending.
2. Pick the highest-confidence box, add it to the final output list.
3. Remove all remaining boxes with `IoU >= threshold` against the picked box (same class).
4. Repeat from step 2 with remaining boxes until none are left.

```python
def nms(boxes, scores, iou_threshold=0.5):
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ious = [iou(boxes[i], boxes[j]) for j in order[1:]]
        order = order[1:][[iou_val < iou_threshold for iou_val in ious]]
    return keep
```

**Soft-NMS variant:** instead of hard-eliminating overlapping boxes, decays their scores proportional to overlap — helps in crowded scenes where two real objects genuinely overlap heavily (e.g., people in a crowd) and hard NMS would wrongly suppress a true detection.

**Pitfalls:**
- Confusing IoU threshold used for *NMS* (suppressing duplicates) with IoU threshold used for *evaluation matching* (deciding TP vs FP) — these are separate, independently tunable thresholds.
- Applying NMS across classes when it should be applied per-class (two different-class objects can legitimately overlap heavily).
- Very low NMS threshold removes legitimate nearby distinct objects; very high threshold leaves duplicate boxes.

---

### Two-stage detectors: R-CNN, Fast R-CNN, Faster R-CNN

**R-CNN (2014).** Pipeline: (1) generate ~2000 region proposals per image using an external algorithm (Selective Search), (2) warp/crop each region and run it independently through a CNN (e.g., AlexNet) to extract features, (3) classify with an SVM per class, (4) regress bounding box coordinates. **Extremely slow** — every proposal requires a full independent CNN forward pass (minutes per image).

**Fast R-CNN (2015).** Key fix: run the CNN **once** over the whole image to get a shared feature map, then use **RoI (Region of Interest) Pooling** to extract a fixed-size feature vector for each region proposal directly from that shared feature map (instead of re-running the CNN per region). Adds a multi-task loss (classification + bounding box regression) trained jointly end-to-end. Much faster than R-CNN (~seconds, not minutes), but region proposals still come from the external Selective Search algorithm (a CPU bottleneck).

**Faster R-CNN (2015).** Replaces the external region-proposal step with a learned **Region Proposal Network (RPN)** — a small CNN head that slides over the shared feature map and, at each location, predicts objectness scores and box refinements for a set of predefined **anchor boxes** (multiple scales/aspect ratios per location). This makes the entire pipeline (backbone → RPN → RoI pooling → classification/regression heads) trainable end-to-end and GPU-accelerated, dramatically increasing speed while improving accuracy.

```
Faster R-CNN pipeline:
Image → CNN backbone (shared feature map)
      → RPN (proposes candidate boxes + objectness score from anchors)
      → RoI Pooling/Align (crop+resize proposal regions from feature map to fixed size)
      → FC head → classification (softmax over classes) + bbox regression (refine coordinates)
```

**RoI Pooling vs RoI Align:** RoI Pooling quantizes/rounds proposal coordinates to the feature map's grid, introducing misalignment (especially harmful for small objects/segmentation). RoI Align (introduced in Mask R-CNN) uses bilinear interpolation to sample feature values at exact (non-quantized) locations, improving localization precision.

**Two-stage tradeoff:** Higher accuracy (especially for small/dense objects) due to the explicit proposal-then-refine structure, at the cost of higher latency compared to one-stage detectors.

---

### One-stage detectors: YOLO, SSD — speed/accuracy tradeoffs

**Core idea of one-stage detection:** skip the separate proposal stage entirely — directly predict class probabilities and bounding box offsets densely across a grid (or set of anchors) in a **single** network forward pass. Much faster, historically at some accuracy cost (especially for small objects), though modern versions (YOLOv7/v8+) have largely closed the gap.

**YOLO (You Only Look Once) — versions overview:**

| Version | Key idea |
|---|---|
| YOLOv1 (2016) | Divides image into an `SxS` grid; each cell directly predicts B boxes + confidence + class probabilities in one pass — real-time speed but struggled with small/overlapping objects |
| YOLOv2/v3 | Added anchor boxes, multi-scale predictions (detecting at 3 feature-map resolutions to better handle objects of different sizes), better backbone (Darknet) |
| YOLOv4/v5 | Engineering improvements: CSPDarknet backbone, Mosaic augmentation, better training tricks (self-adversarial training, various loss improvements) |
| YOLOv6/v7/v8+ | Anchor-free heads, decoupled classification/regression heads, further backbone/loss refinements, better accuracy-speed tradeoff, unified detection/segmentation/pose APIs |

**YOLO grid cell prediction (conceptual):** each grid cell predicts, per anchor: `(tx, ty, tw, th, objectness, class_probs...)`. Box center offsets `(tx, ty)` are passed through a sigmoid so predictions stay within the cell, and width/height `(tw, th)` are predicted as log-space scale factors relative to anchor dimensions:
```
bx = σ(tx) + cx     (cx = grid cell's x-coordinate)
by = σ(ty) + cy
bw = aw · e^(tw)    (aw = anchor width)
bh = ah · e^(th)
```

**SSD (Single Shot MultiBox Detector, 2016):** Similar single-pass philosophy to YOLO, but generates predictions from **multiple feature map layers at different resolutions** simultaneously (early, high-res layers for small objects; later, low-res layers for large objects), each with its own set of anchor boxes ("default boxes"). This multi-scale-by-design approach improved small-object detection relative to YOLOv1.

**Speed/accuracy tradeoff summary:**

| | Two-stage (Faster R-CNN) | One-stage (YOLO/SSD) |
|---|---|---|
| Speed | Slower (multiple stages, RoI-wise computation) | Faster (single dense pass), often real-time |
| Accuracy (esp. small/dense objects) | Historically higher | Historically lower, gap has closed significantly in modern YOLO versions |
| Use case | Offline/high-accuracy needs (medical imaging analysis, satellite imagery) | Real-time applications (autonomous driving perception, video surveillance, robotics) |

---

### Anchor boxes, anchor-free detectors

**Anchor boxes.** Predefined reference boxes of various scales/aspect ratios placed at each spatial location of a feature map. Instead of predicting absolute box coordinates from scratch (a much harder regression problem), the network predicts small **offsets/refinements relative to the nearest-matching anchor**, plus a class/objectness score. Anchors are matched to ground-truth boxes during training via IoU thresholds (e.g., anchor is "positive" if IoU with some GT box > 0.7, "negative" if IoU < 0.3 with all GT boxes, ignored otherwise).

**Why anchors help:** they inject useful priors about typical object shapes/scales (often derived via k-means clustering on the training set's box dimensions, as YOLOv2 did), making the regression target smaller/easier to learn and enabling multiple overlapping objects of different sizes/aspect ratios to be detected at the same spatial location.

**Anchor box drawbacks:**
- Introduces many hyperparameters (number of anchors, scales, aspect ratios) that need dataset-specific tuning.
- Creates severe class imbalance — most anchors are background/negative (addressed by focal loss, hard-negative mining, or careful sampling).
- Anchor matching adds complexity and can be a source of bugs/inefficiency.

**Anchor-free detectors (concept).** Predict objects without predefined anchor boxes, instead directly regressing box properties from key points or dense per-pixel predictions:
- **CenterNet / FCOS-style:** predict object presence and box size/offsets from the *center point* of each object (per-pixel prediction of "is this pixel an object center" + distances to the 4 box edges).
- **Corner-based (CornerNet):** predict paired top-left/bottom-right corner keypoints and group them into boxes.

**Advantages of anchor-free:** simpler design (fewer hyperparameters), no anchor-matching heuristics, often fewer false positives from redundant anchors, and typically comparable or better accuracy with less tuning — this is why modern YOLO versions (v6+) and detectors like FCOS moved toward anchor-free heads.

---

### Evaluation: mAP, IoU thresholds

**Precision & Recall (per class, at an IoU/confidence threshold):**
```
Precision = TP / (TP + FP)
Recall    = TP / (TP + FN)
```
A predicted box is a TP if IoU with some unmatched ground-truth box of the same class ≥ threshold (e.g., 0.5); otherwise FP. Ground-truth boxes with no matching prediction are FN.

**Average Precision (AP)** for one class: the area under the Precision-Recall curve, computed by varying the confidence threshold. Typically computed via interpolation (11-point interpolation in older PASCAL VOC, or all-points interpolation in COCO).

**mean Average Precision (mAP):** average of AP across all classes.
```
mAP = (1/N) * Σ AP_c   for classes c = 1..N
```

**COCO-style mAP (the modern standard):** averages AP across multiple IoU thresholds (0.5 to 0.95, step 0.05 — denoted `mAP@[.5:.95]` or `mAP@[.5,.95]`), giving a stricter, more holistic measure of localization quality than a single fixed IoU threshold. `mAP@0.5` (aka `AP50`, the PASCAL VOC-style metric) is more lenient and commonly reported alongside for interpretability.

```python
# Conceptual sketch of AP computation for one class
def average_precision(pred_boxes_sorted_by_conf, gt_boxes, iou_thresh=0.5):
    tp, fp = [], []
    matched_gt = set()
    for pred in pred_boxes_sorted_by_conf:
        best_iou, best_gt_idx = 0, -1
        for idx, gt in enumerate(gt_boxes):
            if idx in matched_gt:
                continue
            iou_val = iou(pred, gt)
            if iou_val > best_iou:
                best_iou, best_gt_idx = iou_val, idx
        if best_iou >= iou_thresh:
            tp.append(1); fp.append(0)
            matched_gt.add(best_gt_idx)
        else:
            tp.append(0); fp.append(1)
    # then compute cumulative precision/recall and integrate area under curve
    ...
```

**Pitfalls:**
- Reporting only `mAP@0.5` when comparing modern detectors — COCO's stricter `mAP@[.5:.95]` better reflects fine-grained localization quality and is the current standard benchmark.
- Forgetting that mAP averages over classes equally regardless of class frequency — a model can have high mAP while performing terribly on rare classes.
- Not separating detection metrics by object size (COCO also reports AP for small/medium/large objects) — aggregate mAP can hide poor small-object performance.

### Interview Questions

1. **Q: Define IoU and explain its role in both training and evaluation of detectors.**
   A: IoU = intersection area / union area between two boxes, ranging 0 to 1. In training, it's used to match anchors/predictions to ground-truth boxes (defining positive/negative training samples for the loss) and as an optimization target itself (in IoU-based losses like GIoU/DIoU). In evaluation, a prediction is a true positive only if IoU with some ground-truth box (of the correct class) exceeds a chosen threshold (e.g., 0.5), directly determining precision/recall/mAP.

2. **Q: Walk through the Non-Max Suppression algorithm and explain why it's necessary.**
   A: Detectors typically output many overlapping candidate boxes per real object (from dense grid cells/anchors). NMS: sort by confidence, greedily keep the highest-confidence box, discard all remaining boxes with IoU above a threshold against it (same class), repeat until no boxes remain. It's necessary to collapse redundant detections into one clean prediction per object.

3. **Q: What is the fundamental architectural difference between Fast R-CNN and Faster R-CNN?**
   A: Fast R-CNN still relies on an external, non-learned region proposal algorithm (Selective Search) to generate candidate boxes, which is a CPU bottleneck and not trainable end-to-end. Faster R-CNN replaces this with a learned Region Proposal Network (RPN) that shares the backbone's feature map and predicts proposals directly via anchor-based regression/classification, making the entire pipeline trainable end-to-end on GPU and much faster.

4. **Q: Explain RoI Pooling and why RoI Align improves on it.**
   A: RoI Pooling crops a region proposal from the shared feature map and pools it into a fixed spatial size, but it quantizes/rounds the proposal's coordinates and pooling bin boundaries to the feature map's integer grid, causing small misalignments between the proposal and the extracted features — especially damaging for small objects and pixel-precise tasks like segmentation. RoI Align avoids this by using bilinear interpolation to sample feature values at the exact (fractional) coordinates, preserving spatial precision.

5. **Q: Compare the speed/accuracy tradeoff between YOLO and Faster R-CNN, and explain the architectural reason for the difference.**
   A: YOLO processes the entire image in a single forward pass, directly predicting boxes/classes densely across a grid — no separate proposal stage — making it much faster and suitable for real-time use. Faster R-CNN's two-stage design (region proposals, then per-region refinement/classification) does more targeted, iterative computation per candidate region, historically yielding higher accuracy, especially on small or overlapping objects, at the cost of speed. Modern YOLO versions have significantly closed this accuracy gap through better backbones, multi-scale prediction, and loss improvements.

6. **Q: What problem do anchor boxes solve, and what's a major drawback of using them?**
   A: Anchors give the network shape/scale priors so it predicts small relative offsets rather than absolute box coordinates from scratch, and allow multiple objects of different sizes/aspect ratios to be detected at the same spatial location. Drawback: they introduce many hyperparameters (number, scales, aspect ratios) requiring dataset-specific tuning, and cause severe foreground/background class imbalance since most anchors don't correspond to any real object.

7. **Q: How do anchor-free detectors like FCOS or CenterNet localize objects without predefined anchor boxes?**
   A: They predict, at each spatial location (typically the object's center), whether an object center exists there and directly regress the distances/offsets to the box's edges (or corners), turning detection into a per-pixel dense prediction + regression problem rather than an anchor-matching problem — simplifying the pipeline and removing anchor-related hyperparameters.

8. **Q: How is mAP computed, and why does COCO use mAP averaged across IoU thresholds 0.5 to 0.95 rather than a single threshold?**
   A: For each class, compute the precision-recall curve by varying the confidence threshold (using IoU ≥ some threshold to decide TP/FP), then take the area under that curve (AP); mAP averages AP across all classes. COCO averages this across 10 IoU thresholds (0.5 to 0.95 step 0.05) because a single lenient threshold (like 0.5) doesn't penalize imprecise localization — a stricter, threshold-averaged metric better rewards precise bounding-box localization, not just rough overlap.

9. **Q: In a scene with a large crowd of overlapping people, hard NMS might suppress correct detections. How would you address this?**
   A: Use Soft-NMS, which decays (rather than zeroes out) the confidence score of overlapping boxes proportional to their IoU with the kept box, rather than hard-eliminating them — allowing genuinely distinct, heavily-overlapping objects to still surface as separate detections if their scores remain above the final confidence threshold. Alternatively, use a smaller/class-specific NMS IoU threshold or a detector explicitly designed for crowded scenes.

10. **Q: Explain the multi-task loss used in Faster R-CNN / Fast R-CNN.**
    A: A combined loss `L = L_cls + λ·L_box`, where `L_cls` is typically a (multi-class) cross-entropy/log loss over the RoI's predicted class distribution, and `L_box` is a smooth-L1 (Huber) regression loss over the 4 box coordinate offsets, applied only for positive (foreground-matched) samples. This joint training lets the shared backbone features be optimized simultaneously for both classification and precise localization.

11. **Q: Why does SSD predict from multiple feature map layers instead of just the final one?**
    A: Earlier, higher-resolution feature maps have smaller receptive fields and finer spatial detail, better suited for detecting small objects; later, lower-resolution feature maps have larger receptive fields, better suited for detecting large objects. Predicting from multiple scales lets a single-shot detector natively handle multi-scale objects without needing an explicit image pyramid at inference time.

12. **Q: What is "hard negative mining" and why is it needed in object detection training?**
    A: Because the vast majority of anchors/regions in any image are background (easy negatives), naively training on all of them would let easy negatives dominate the loss and drown out the useful signal from the rare hard negatives (background regions that look similar to objects) and positives. Hard negative mining selects the highest-loss (most confusable) negative examples to include in each training batch (or uses focal loss to down-weight easy negatives automatically), improving learning efficiency and reducing false positives.

13. **Q: A detector achieves 90% mAP@0.5 but only 60% mAP@[.5:.95]. What does this tell you?**
    A: The model is good at roughly localizing objects (finding the right region) but not precisely fitting tight bounding boxes — its boxes have significant offset/scale error, which only shows up under stricter IoU thresholds (0.6–0.95). This suggests focusing on improving box regression quality (better regression loss, e.g., GIoU/DIoU/CIoU loss, or RoI Align-style precision) rather than detection/classification per se.

14. **Q: How would you adapt a general-purpose object detector to detect very small objects (e.g., tiny defects in manufacturing images)?**
    A: Use higher input resolution (avoid excessive downsampling that erases small objects), leverage/emphasize higher-resolution early feature maps (as in SSD/FPN-style multi-scale detection), use smaller anchor scales tuned to your object size distribution (or an anchor-free approach that doesn't need scale priors), apply tiling/cropping of large images into smaller patches for inference, and use appropriate augmentation (e.g., avoid heavy downsampling augmentations) and IoU thresholds tuned to smaller box sizes since IoU is inherently more sensitive to small localization errors on small objects.

15. **Q: What's the role of Feature Pyramid Networks (FPN) in modern detectors, and how does it relate to the multi-scale prediction idea in SSD?**
    A: FPN builds a top-down pathway with lateral connections that combines high-resolution, low-semantic early features with low-resolution, high-semantic late features at every pyramid level, so *every* scale of feature map benefits from strong semantic information (not just the deepest layer, as in a naive backbone). This gives better multi-scale detection than SSD's simpler approach of predicting directly from unenhanced backbone layers at different depths, and is now a near-universal component in modern detectors (Faster R-CNN+FPN, RetinaNet, YOLO variants).

---

## Image Segmentation

### Semantic vs instance vs panoptic segmentation

| Type | What it predicts | Distinguishes instances of same class? | Example output |
|---|---|---|---|
| **Semantic segmentation** | Per-pixel class label | No — all "car" pixels get one label, regardless of how many cars | "This pixel = road, this pixel = car" |
| **Instance segmentation** | Per-pixel class label + separate mask per object instance | Yes — each individual object gets a distinct mask/ID | "Car #1 mask, Car #2 mask, ..." |
| **Panoptic segmentation** | Unified: semantic labels for background/"stuff" (sky, road) + instance masks for countable "things" (cars, people) | Yes, for "thing" classes; "stuff" classes are just semantic | Combines both into one coherent output covering every pixel |

**Intuition:** Semantic segmentation answers "what is at each pixel," instance segmentation additionally answers "which specific object," and panoptic segmentation unifies both so that every pixel in the image is accounted for exactly once (no pixel is left unlabeled, and no pixel belongs to two different masks) — a requirement neither pure semantic nor pure instance segmentation enforces on its own.

---

### U-Net architecture (encoder-decoder with skip connections)

**Motivation.** Segmentation requires dense, pixel-level, high-resolution output — but standard CNN classification architectures progressively downsample, losing spatial resolution. U-Net solves this with a symmetric **encoder-decoder** structure:

- **Encoder (contracting path):** a series of conv blocks + downsampling (max-pool), progressively extracting increasingly abstract, semantic features while reducing spatial resolution — much like a standard CNN classifier backbone.
- **Decoder (expanding path):** a series of upsampling (transposed convolution or interpolation + conv) + conv blocks, progressively restoring spatial resolution back to the input size.
- **Skip connections:** at each resolution level, the encoder's feature map is concatenated with the corresponding decoder feature map *before* further processing. This reinjects fine-grained spatial detail (edges, precise boundaries) that would otherwise be lost during downsampling, which the decoder alone (working only from the compressed bottleneck) cannot recover.

```
Input image
   │
Encoder: Conv-Conv-Pool (×N, channels double, resolution halves each stage)
   │                                    │  (skip connections at each stage)
   ▼                                    ▼
Bottleneck (most abstract features)     │
   │                                    │
Decoder: Upsample-Concat[skip]-Conv-Conv (×N, channels halve, resolution doubles each stage)
   │
Output segmentation mask (same H x W as input, C = num_classes channels)
```

```python
import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.block(x)

class UNet(nn.Module):
    def __init__(self, in_ch=3, num_classes=1):
        super().__init__()
        self.enc1 = DoubleConv(in_ch, 64)
        self.enc2 = DoubleConv(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(128, 256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = DoubleConv(256, 128)   # 256 = 128 (upsampled) + 128 (skip)
        self.up1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = DoubleConv(128, 64)    # 128 = 64 (upsampled) + 64 (skip)
        self.out_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1)
```

**Why skip connections matter:** without them, the decoder must reconstruct fine boundaries purely from the heavily compressed bottleneck representation, resulting in blurry, imprecise masks — analogous to ResNet's residual connections, skip connections here preserve high-resolution spatial information that would otherwise be destroyed by pooling.

**Common uses:** medical image segmentation (U-Net's original domain — cell/organ/tumor segmentation), satellite imagery, any dense pixel-labeling task, especially where training data is limited (U-Net's efficient design and heavy use of augmentation made it effective even with few hundred/thousand labeled images).

---

### Mask R-CNN for instance segmentation

**Core idea:** extend Faster R-CNN (object detection) with an additional **mask prediction branch**. For each detected RoI (region of interest), in parallel with the existing classification and box-regression heads, a small FCN (fully convolutional network) head predicts a binary segmentation mask **per class**, at a fixed small resolution (e.g., 28x28), which is then resized back to the RoI's actual size.

```
Image → Backbone (+FPN) → RPN → RoI Align → ┬─ classification head → class
                                             ├─ box regression head → refined box
                                             └─ mask head (small FCN) → per-class binary mask (e.g., 28x28)
```

**Why RoI Align (not RoI Pooling) matters here specifically:** since masks require pixel-precise spatial alignment, the quantization error introduced by RoI Pooling is especially damaging for mask quality — this was a key motivation for introducing RoI Align in the Mask R-CNN paper.

**Loss:** multi-task loss `L = L_cls + L_box + L_mask`, where `L_mask` is a per-pixel binary cross-entropy loss, applied only to the mask corresponding to the RoI's ground-truth class (masks for other classes don't contribute to the loss for that RoI — this avoids inter-class competition in the mask branch).

**Pitfalls:**
- Confusing Mask R-CNN's mask branch resolution (small, fixed, e.g., 28x28) with the final output resolution — the small mask is resized/pasted back into the full-resolution box region at inference.
- Forgetting Mask R-CNN still relies on a detection backbone (RPN) — its instance segmentation quality is bounded by detection quality; missed/wrong detections mean missed/wrong masks.

---

### Evaluation: IoU, Dice coefficient, pixel accuracy

**Pixel accuracy:** simplest metric — fraction of correctly classified pixels overall.
```
Pixel Accuracy = (correctly classified pixels) / (total pixels)
```
**Major pitfall:** highly misleading under class imbalance — e.g., if 95% of pixels are background, a trivial "always predict background" model scores 95% pixel accuracy while being useless for the foreground class.

**IoU (Jaccard Index) for segmentation:** same formula as detection IoU but applied to pixel sets/masks rather than boxes:
```
IoU = |Prediction ∩ Ground Truth| / |Prediction ∪ Ground Truth|
```
Often reported as **mean IoU (mIoU)**: average IoU across all classes, treating each class equally regardless of pixel frequency — more robust to class imbalance than pixel accuracy.

**Dice coefficient (F1-score for segmentation):**
```
Dice = 2·|Prediction ∩ Ground Truth| / (|Prediction| + |Ground Truth|)
```
Relationship to IoU: `Dice = 2·IoU / (1 + IoU)` — Dice is always ≥ IoU, and both move together monotonically, but Dice weights overlap more heavily (less harshly penalizes partial mismatches) than IoU. Dice is extremely common in medical image segmentation, often used directly as a **loss function** (Dice Loss = `1 - Dice`) because it handles class imbalance (small foreground structures like tumors) better than plain pixel-wise cross-entropy.

```python
def dice_coefficient(pred_mask, gt_mask, eps=1e-6):
    intersection = (pred_mask * gt_mask).sum()
    return (2. * intersection + eps) / (pred_mask.sum() + gt_mask.sum() + eps)

def iou_score(pred_mask, gt_mask, eps=1e-6):
    intersection = (pred_mask * gt_mask).sum()
    union = pred_mask.sum() + gt_mask.sum() - intersection
    return (intersection + eps) / (union + eps)
```

**Pitfalls:**
- Using plain pixel accuracy as the primary metric for imbalanced segmentation tasks (e.g., tumor segmentation where the tumor is <1% of pixels) — always prefer mIoU/Dice.
- Combining Dice Loss with cross-entropy incorrectly (common practical remedy: use a weighted sum `L = L_CE + λ·L_Dice` to get both stable gradients from CE early in training and better handling of class imbalance/boundary quality from Dice).

### Interview Questions

1. **Q: Explain the difference between semantic, instance, and panoptic segmentation with an example.**
   A: Semantic segmentation labels every pixel by class only — e.g., all pixels belonging to any car get the label "car," with no distinction between individual cars. Instance segmentation additionally separates individual object instances — "car #1," "car #2," each with its own mask. Panoptic segmentation unifies both: it assigns semantic labels to background/uncountable "stuff" (sky, road) and separate instance masks to countable "thing" classes (cars, people), covering every pixel in the image exactly once.

2. **Q: Why can't a standard CNN classifier architecture be used directly for segmentation without modification?**
   A: Classifier CNNs progressively downsample spatial resolution (via pooling/stride) to build abstract, low-resolution semantic representations suitable for a single whole-image label. Segmentation requires a dense, per-pixel, full-resolution output, so the architecture needs a decoder/upsampling path to recover spatial resolution — this is exactly what encoder-decoder architectures like U-Net or fully convolutional networks (FCN) provide.

3. **Q: What is the purpose of skip connections in U-Net, and what would happen if you removed them?**
   A: Skip connections concatenate high-resolution encoder feature maps directly into the corresponding decoder stage, reinjecting fine spatial detail (edges, precise object boundaries) lost during downsampling. Without them, the decoder must reconstruct all spatial detail purely from the heavily compressed bottleneck representation, leading to blurry segmentation masks with imprecise boundaries.

4. **Q: How does Mask R-CNN extend Faster R-CNN to perform instance segmentation?**
   A: It adds a third parallel branch (alongside classification and box regression) to each RoI: a small FCN "mask head" that predicts a per-class binary segmentation mask at a fixed small resolution (e.g., 28x28), which is resized to fit the RoI's actual box dimensions at inference. It also replaces RoI Pooling with RoI Align to preserve the spatial precision needed for pixel-accurate masks.

5. **Q: Why is pixel accuracy a poor metric for evaluating segmentation models on imbalanced datasets, and what should be used instead?**
   A: Pixel accuracy is dominated by the majority class — if background occupies 95%+ of pixels, a model that never correctly segments the minority foreground class can still score very high pixel accuracy. Mean IoU (mIoU) or Dice coefficient, computed and averaged per-class, are preferred because they weight all classes (including rare/small ones) equally rather than by pixel frequency.

6. **Q: Derive the relationship between IoU and Dice coefficient.**
   A: Given intersection `I` and union `U = A + B - I` (where A, B are the areas/pixel counts of the two masks): `IoU = I/U = I/(A+B-I)`, and `Dice = 2I/(A+B)`. Solving for the relationship: `Dice = 2·IoU / (1 + IoU)`. This shows Dice is always ≥ IoU for IoU in [0,1], and both are monotonically related but Dice is less harsh on partial overlaps.

7. **Q: Why is Dice Loss often preferred over cross-entropy loss for medical image segmentation?**
   A: Medical segmentation targets (e.g., tumors, lesions) are often tiny relative to the whole image, causing severe class imbalance. Pixel-wise cross-entropy treats every pixel equally and gets dominated by the easy, abundant background class, providing weak gradient signal for the rare foreground. Dice Loss directly optimizes the overlap ratio between prediction and ground truth, which is far less sensitive to the absolute class-frequency imbalance, producing better-calibrated gradients toward improving foreground segmentation quality.

8. **Q: What is the tradeoff of using a small fixed mask resolution (e.g., 28x28) in Mask R-CNN's mask head?**
   A: It keeps the mask head lightweight/fast and decouples mask resolution from RoI size, but limits the fineness of mask boundary detail, especially for large or highly detailed objects — after upsampling the small mask to the RoI's actual size, fine boundary detail can appear blocky/imprecise compared to methods that predict masks at higher native resolution.

9. **Q: How would you adapt a segmentation pipeline for a panoptic segmentation task where you already have separate semantic and instance segmentation models?**
   A: Merge outputs by first taking instance segmentation results for "thing" classes (countable objects) with high-confidence masks, then filling in remaining unlabeled pixels using the semantic segmentation model's predictions for "stuff" classes (background/uncountable regions), resolving overlaps (e.g., by instance confidence and non-overlap constraints) so every pixel is assigned exactly one label — this fusion approach was used by early panoptic segmentation baselines before end-to-end panoptic architectures (e.g., Panoptic FPN) were developed.

10. **Q: Why does U-Net use concatenation for skip connections while ResNet uses addition for its skip connections? What's the practical implication?**
    A: ResNet's addition requires matching channel dimensions and represents "refining an existing signal" (residual learning) — the identity and the residual live in the same feature space. U-Net's concatenation preserves both the decoder's own upsampled features and the encoder's original features as distinct channels, allowing the subsequent convolution to learn how to combine coarse semantic information and fine spatial detail flexibly, rather than forcing them into a shared additive space — practically, concatenation increases channel count (and thus computation) at each decoder stage, unlike addition.

11. **Q: What is a fully convolutional network (FCN) and why was it a milestone for segmentation?**
    A: An FCN replaces the final fully-connected classification layers of a CNN with convolutional layers (including 1x1 convs) and upsampling, so the entire network is convolutional end-to-end and can produce a dense, per-pixel output map instead of a single class label — this was the first major architecture demonstrating that a classifier backbone could be repurposed, with a learned upsampling path, into an effective, trainable end-to-end segmentation model.

12. **Q: You're building a defect segmentation model where defects average 0.5% of pixels per image. What loss/metric choices would you make?**
    A: Use a combined loss like weighted cross-entropy (with high weight on the rare defect class) plus Dice Loss (or Focal Loss variants) to counteract imbalance; evaluate with per-class IoU/Dice rather than pixel accuracy; consider patch-based training/sampling that oversamples defect-containing regions; and monitor false-negative rate carefully since missing a real defect is often more costly than a false positive in quality-control settings.

13. **Q: What's the difference between transposed convolution and simple bilinear upsampling + convolution in a decoder, and which would you choose?**
    A: Transposed convolution learns the upsampling kernel jointly with the rest of the network (more flexible, but prone to "checkerboard" artifacts if not carefully configured — e.g., stride/kernel size mismatches). Bilinear upsampling followed by a regular convolution first upsamples deterministically (no learned artifacts) then lets a normal conv refine features — often more stable and is a common practical choice to avoid checkerboard patterns, at a small cost in flexibility versus a fully learned upsampling operator.

14. **Q: How do you evaluate an instance segmentation model differently from a semantic segmentation model?**
    A: Instance segmentation is evaluated similarly to object detection but with mask IoU instead of box IoU — computing mask-based mAP (e.g., COCO's mask AP) across IoU thresholds, matching predicted instance masks to ground-truth instance masks. Semantic segmentation is evaluated per-pixel across the whole image without any notion of "matching individual instances," typically via mIoU or Dice per class.

15. **Q: If your U-Net segmentation model produces good IoU on training data but poor boundary precision on validation images, what would you try?**
    A: Check for overfitting (regularization, more augmentation — especially elastic deformations/random crops relevant to the boundary-sensitive domain); consider boundary-aware loss terms (e.g., boundary loss, or adding a weighted edge-pixel term to the loss); verify sufficient decoder capacity/skip-connection usage to preserve fine detail; check whether the input resolution is sufficient to represent fine boundaries; and inspect for label noise/inconsistent annotation quality at boundaries, a very common real-world cause of boundary imprecision.

---

## Vision Transformers and Modern Architectures

### ViT: patch embedding, positional encoding for images, how attention replaces convolution

**Core idea (Vision Transformer, ViT, 2020):** treat an image as a sequence of flattened patches, analogous to tokens in NLP, and apply a standard Transformer encoder directly, with essentially no convolution-specific inductive bias.

**Patch embedding:**
1. Split the input image (e.g., `224x224x3`) into fixed-size non-overlapping patches (e.g., `16x16`), yielding `(224/16)² = 196` patches.
2. Flatten each patch into a vector (`16*16*3 = 768` values) and linearly project it into the model's embedding dimension `D` via a learned matrix — equivalent to a single strided convolution with kernel=stride=patch size.
3. Prepend a learnable `[CLS]` token (as in BERT) whose final representation is used for classification.

**Positional encoding for images:** since self-attention is permutation-invariant (it has no inherent notion of spatial order), a **learned positional embedding** is added to each patch embedding (ViT uses simple learned 1D positional embeddings indexed by patch position in the flattened sequence, rather than a fixed sinusoidal or explicitly 2D encoding — surprisingly, this simple approach works well because the model learns 2D spatial relationships from the data).

```
Image (H x W x 3)
   → split into N patches of size P x P
   → flatten + linear projection → patch embeddings (N x D)
   → prepend [CLS] token → (N+1) x D
   → add learned positional embeddings
   → standard Transformer encoder (multi-head self-attention + MLP blocks, x L layers)
   → take [CLS] token's final representation → classification head
```

```python
import torch
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_ch=3, embed_dim=768):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        # a single conv with kernel=stride=patch_size == "split into patches + linear project"
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                       # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)        # (B, num_patches, embed_dim)
        return x

class SimpleViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=768, depth=12, num_heads=12, num_classes=1000):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_tokens, x], dim=1) + self.pos_embed
        x = self.encoder(x)
        return self.head(x[:, 0])   # use CLS token representation
```

**How attention replaces convolution:** self-attention computes, for every patch, a weighted aggregation of *all other patches* (global receptive field from layer 1), where weights are learned dynamically based on content similarity (query-key dot products), rather than a fixed, local, weight-shared kernel. This means:
- Convolution has a strong **spatial locality + translation-equivariance inductive bias** built into its architecture; attention has none built in — it must *learn* any useful spatial structure/locality purely from data.
- Attention's receptive field is global immediately (every token attends to every other token), whereas CNN receptive field grows only gradually with depth.

---

### Comparison: CNN inductive bias vs transformer flexibility, data requirements

| Aspect | CNN | ViT / Transformer |
|---|---|---|
| Inductive bias | Strong: locality, translation equivariance, hierarchical composition | Weak/none: must learn spatial relationships from data |
| Receptive field | Grows gradually with depth | Global from the first layer |
| Data efficiency | Good with small/medium datasets (bias helps generalize) | Needs large-scale pretraining data (ImageNet-21k, JFT-300M) to match/exceed CNNs — underperforms CNNs when trained from scratch on small datasets |
| Scalability | Accuracy tends to plateau with more data/compute | Scales very well with more data/compute/model size (empirically, "more data + more compute → better ViT," a key motivation for its adoption) |
| Compute pattern | Efficient, hardware-friendly convolutions | Quadratic self-attention cost in number of patches (`O(N²)`) |

**Why ViT needs more data:** without built-in locality/translation-equivariance priors, ViT must discover from scratch (via training) that nearby pixels are related and that objects look the same regardless of position — a CNN gets this "for free" from its architecture. With enough data, the Transformer's greater flexibility (no restrictive prior) lets it discover potentially better task-specific structure than a fixed convolutional prior, but with too little data, it overfits/underperforms relative to a CNN that has useful priors baked in.

**Practical resolution:** hybrid architectures (e.g., convolutional stem + transformer body), or heavy pretraining (self-supervised or supervised) on massive datasets before fine-tuning on the target task, are the standard ways to get ViT's benefits without needing an impractically large labeled target dataset.

---

### Self-supervised vision pretraining concepts

**Why self-supervised pretraining matters:** labeled data is expensive; self-supervised learning (SSL) lets models learn useful representations from large amounts of *unlabeled* images by solving an automatically-generated ("pretext") task, then transfer those representations to downstream tasks (classification, detection, segmentation) via fine-tuning or linear probing.

**Contrastive learning — SimCLR (concept):**
- Take an image, generate two different augmented views of it (random crop, color jitter, flip, etc.) — these are a "positive pair."
- Views from *different* images in the batch are treated as "negative pairs."
- Train an encoder + projection head so that positive pairs are pulled close together in embedding space while negative pairs are pushed apart, using a contrastive loss (NT-Xent / InfoNCE):
```
L = -log[ exp(sim(z_i, z_j)/τ) / Σ_k≠i exp(sim(z_i, z_k)/τ) ]
```
where `sim` is cosine similarity, `τ` is a temperature hyperparameter, `z_i, z_j` are the two augmented views' embeddings.
- Intuition: the model is forced to learn representations invariant to the augmentations applied (since both views of the same image must map close together) while remaining discriminative enough to distinguish different images (since different images' views must map apart) — this yields features that transfer well to downstream tasks.
- Requires **large batch sizes** (many negatives per positive pair) for strong performance, which is computationally expensive.

**Contrastive learning — MoCo (concept):**
- Addresses SimCLR's large-batch requirement by maintaining a **momentum encoder** and a large **queue of negative embeddings** from previous batches (a dynamic dictionary), decoupling the number of negatives from the batch size.
- The momentum encoder's weights are an exponential moving average of the main encoder's weights, providing more consistent (slowly evolving) representations for the negative queue, improving training stability.

**Masked Image Modeling — MAE (Masked Autoencoders, concept):**
- Inspired by BERT's masked language modeling: randomly mask a large fraction (e.g., 75%) of image patches, and train an encoder-decoder to reconstruct the missing patches' raw pixel values from only the visible patches.
- Key efficiency insight: the (large) encoder only processes the small set of *visible* patches (not the masked ones), while a lightweight decoder reconstructs the full image from encoded visible patches + mask tokens — this asymmetric design makes MAE pretraining computationally efficient despite a high masking ratio.
- Unlike contrastive methods, MAE requires no negative pairs, no large batch sizes, and no careful augmentation pair design — just masked reconstruction, making it simpler to scale.

| Method | Task | Key requirement | Efficiency note |
|---|---|---|---|
| SimCLR | Contrastive (pull positive pairs together, push negatives apart) | Large batch size (many negatives) | Simple, but batch-size-hungry |
| MoCo | Contrastive with momentum encoder + memory queue | Momentum update + queue | Decouples negatives from batch size |
| MAE | Masked patch reconstruction (generative) | High masking ratio (~75%) | Encoder only sees visible patches — very compute-efficient |

---

### Multimodal vision-language models (CLIP)

**CLIP (Contrastive Language-Image Pretraining):** jointly trains an image encoder (ViT or ResNet) and a text encoder (Transformer) to embed images and their corresponding natural-language captions **into the same shared embedding space**, using a contrastive objective over large-scale (image, caption) pairs scraped from the internet (400M pairs in the original paper).

**Training objective:** for a batch of `N` (image, text) pairs, compute the `N x N` matrix of cosine similarities between all image embeddings and all text embeddings. The loss (symmetric InfoNCE) encourages the similarity between each image and its *true* matching caption to be high, while similarity with all other (mismatched) captions in the batch is low — and symmetrically for text-to-image.

```
image embeddings: I_1...I_N     text embeddings: T_1...T_N
similarity matrix S[i][j] = cosine_sim(I_i, T_j) / τ
Loss = (CrossEntropy(S, diagonal_labels, axis=image) + CrossEntropy(S, diagonal_labels, axis=text)) / 2
```

**Why this is powerful — zero-shot classification:** once trained, CLIP can classify an image into *any* set of user-defined classes at inference time without any task-specific fine-tuning: embed the image, embed text prompts like `"a photo of a {class_name}"` for each candidate class, and pick the class whose text embedding has the highest cosine similarity to the image embedding. This is a major departure from traditional classifiers fixed to a predefined label set.

```python
# Conceptual usage (using a HuggingFace-style CLIP model)
import torch
from transformers import CLIPModel, CLIPProcessor

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

candidate_labels = ["a photo of a cat", "a photo of a dog", "a photo of a car"]
inputs = processor(text=candidate_labels, images=my_image, return_tensors="pt", padding=True)
outputs = model(**inputs)
probs = outputs.logits_per_image.softmax(dim=1)   # zero-shot classification probabilities
```

**Downstream impact:** CLIP embeddings/objective underpin many later systems — text-to-image generation guidance (early DALL-E/Stable Diffusion used CLIP-style guidance), image-text retrieval systems, and as a general-purpose vision backbone for multimodal LLMs (feeding CLIP's visual embeddings into an LLM's input space, as in models like LLaVA-style vision-language assistants).

**Pitfalls:**
- Assuming ViT is a strict replacement for CNNs in all settings — for small labeled datasets or latency-constrained deployment without pretrained weights, CNNs (or hybrid architectures) often remain more practical.
- Forgetting the quadratic self-attention cost — very high-resolution images or long patch sequences make plain ViT computationally expensive; this motivated windowed/hierarchical attention variants (e.g., Swin Transformer).
- Assuming CLIP "understands" images the way humans do — its zero-shot performance is highly sensitive to prompt phrasing ("prompt engineering" matters even for vision-language contrastive models) and inherits biases from its noisy, internet-scraped training data.

### Interview Questions

1. **Q: Explain how an image is converted into a sequence for input to a Vision Transformer.**
   A: The image is split into fixed-size non-overlapping patches (e.g., 16x16 pixels), each patch is flattened into a vector and linearly projected into the model's embedding dimension (equivalent to a single conv with kernel size = stride = patch size), a learnable `[CLS]` token is prepended, and learned positional embeddings are added to encode patch order/location before feeding the sequence into a standard Transformer encoder.

2. **Q: Why does self-attention need positional encodings when convolution doesn't need anything analogous?**
   A: Self-attention computes a weighted combination over all tokens without any inherent sense of their spatial arrangement — it's permutation-invariant, so shuffling the patches would produce shuffled-but-otherwise-identical output. Convolution inherently encodes position implicitly via the fixed spatial arrangement of the sliding kernel operation itself, so no additional encoding is needed. ViT needs explicit positional embeddings added to patch embeddings to inject spatial order information.

3. **Q: Why do ViTs generally need more training data than CNNs to reach comparable accuracy?**
   A: CNNs have strong built-in inductive biases (locality, translation equivariance) that constrain the hypothesis space toward solutions that are naturally well-suited to images, helping them generalize well even with limited data. ViTs lack these built-in biases and must learn any useful spatial structure entirely from data, requiring much larger training sets (or large-scale pretraining) to reach — and eventually surpass — CNN-level performance.

4. **Q: What is the computational complexity of self-attention with respect to the number of patches, and why does this matter for high-resolution images?**
   A: Self-attention has `O(N²)` complexity in the number of tokens/patches `N`, since every token attends to every other token. For high-resolution images, more/smaller patches are needed to preserve detail, so `N` grows quickly and the quadratic cost becomes a serious bottleneck — motivating hierarchical/windowed attention architectures like Swin Transformer that restrict attention to local windows with periodic cross-window connections.

5. **Q: Describe the SimCLR contrastive learning framework and its loss function intuition.**
   A: SimCLR generates two augmented views of each image in a batch (a positive pair) and treats all other images' views in the batch as negatives. It trains an encoder + projection head with the NT-Xent/InfoNCE loss to maximize cosine similarity between positive pairs' embeddings relative to all negative pairs, scaled by a temperature parameter. This forces the encoder to learn representations invariant to the applied augmentations while remaining discriminative across different images, without requiring any labels.

6. **Q: How does MoCo address a key limitation of SimCLR?**
   A: SimCLR needs very large batch sizes to have enough in-batch negative examples for effective contrastive learning, which is computationally expensive. MoCo decouples the number of negatives from batch size by maintaining a queue of negative embeddings from recent past batches and a momentum-updated encoder (an exponential moving average of the main encoder) to keep the queue's embeddings consistent over time, enabling effective contrastive learning with much smaller batch sizes.

7. **Q: Explain the Masked Autoencoder (MAE) pretraining approach and why its encoder-decoder design is asymmetric.**
   A: MAE randomly masks a large fraction (e.g., 75%) of image patches and trains the model to reconstruct the missing patches' pixels from the remaining visible ones. The design is asymmetric because the (larger, more expensive) encoder only processes the small set of visible patches, while a lightweight decoder handles reconstructing the full image (visible tokens + learnable mask tokens) — this makes pretraining much more compute-efficient than processing all patches (visible and masked) through the full encoder.

8. **Q: How does CLIP's training objective differ from a standard image classification objective?**
   A: Standard classification trains an encoder against a fixed, predefined set of class labels using cross-entropy on class logits. CLIP jointly trains an image encoder and text encoder with a contrastive objective over (image, caption) pairs — no fixed label set at all — pulling matching image-text pairs' embeddings together and pushing mismatched pairs apart in a shared embedding space, which enables flexible zero-shot classification against arbitrary text-defined classes at inference time.

9. **Q: How would CLIP classify an image into classes it never saw explicit training examples for (zero-shot)?**
   A: Embed the image using CLIP's image encoder; construct text prompts for each candidate class (e.g., "a photo of a {class}"); embed each prompt with CLIP's text encoder; compute cosine similarity between the image embedding and each text embedding; the class with the highest similarity is the predicted label — no gradient updates or fine-tuning needed for new classes, only new text prompts.

10. **Q: What's a key practical difference between contrastive pretraining (SimCLR/MoCo/CLIP) and masked-modeling pretraining (MAE) in terms of what "negative examples" mean?**
    A: Contrastive methods explicitly require negative examples (other images/captions in the batch or queue) that the model must push away from the positive pair — this makes batch composition and size an important design/hyperparameter concern. MAE is purely generative/reconstructive (predicting masked pixels from context) and needs no negative examples at all, simplifying training dynamics and removing batch-size sensitivity tied to negative sampling.

11. **Q: In what scenario would you choose a CNN over a ViT for a new vision project?**
    A: When labeled/pretraining data is limited and no suitable large-scale pretrained ViT checkpoint exists for the domain, when latency/resource constraints favor architectures with well-optimized, mature efficient variants (MobileNet-class), or when the task strongly benefits from built-in locality/translation-equivariance priors that would otherwise need to be learned from scarce data — CNNs remain the safer, more data-efficient default in these cases.

12. **Q: Why is the receptive field of a single self-attention layer in ViT considered "global," and what's the practical consequence?**
    A: Every patch token computes attention weights over every other patch token in the sequence in a single layer, so information from the entire image can influence any output token after just one layer — contrasted with CNNs where receptive field grows gradually with depth. Practically, this means ViT can capture long-range dependencies/global context very early, which can help with tasks requiring holistic image understanding, but also means it lacks a built-in preference for local/short-range structure that many visual patterns naturally exhibit.

13. **Q: What downstream applications rely on CLIP-style embeddings beyond simple zero-shot classification?**
    A: Text-to-image generation guidance (early diffusion/GAN systems used CLIP similarity as a guidance signal), image-text retrieval systems (searching images by text query or vice versa), content moderation/similarity search, and as a vision "connector" module feeding visual embeddings into multimodal large language models (vision-language assistants) so the LLM can reason jointly over image and text inputs.

14. **Q: What's a known limitation or failure mode of CLIP-style zero-shot classification?**
    A: Sensitivity to prompt phrasing (different wordings of the same class concept can yield notably different similarity scores — "prompt engineering" matters), inherited biases from noisy internet-scraped (image, caption) training data, difficulty with fine-grained distinctions not well captured by typical web captions (e.g., distinguishing similar-looking species/models), and weaker performance on tasks requiring precise spatial/counting reasoning, which contrastive image-text alignment doesn't explicitly optimize for.

15. **Q: How would you combine self-supervised pretraining with a small labeled dataset for a specialized domain (e.g., satellite imagery) where ImageNet pretraining transfers poorly?**
    A: Pretrain a model (CNN or ViT) using self-supervised learning (e.g., MAE or SimCLR/MoCo) directly on a large corpus of *unlabeled* in-domain satellite imagery (which is often much more available than labeled data), then fine-tune this domain-adapted pretrained model on the small labeled dataset for the target task — this typically transfers much better than generic ImageNet pretraining because the self-supervised pretraining has already learned domain-relevant low- and mid-level visual statistics.

---

## Video, Face, OCR, Pose, and 3D Vision

### Optical flow and action recognition (video understanding)

**Optical flow** estimates the apparent 2D motion field `(u, v)` of pixels between consecutive video frames, assuming **brightness constancy**: a point's intensity is preserved as it moves.

```
I(x, y, t) = I(x + u·dt, y + v·dt, t + dt)
```

Taylor-expanding the right side and dropping higher-order terms gives the **optical flow constraint equation**:

```
Ix·u + Iy·v + It = 0
```

This is one equation with two unknowns per pixel — the classic **aperture problem** (motion along a locally uniform edge/texture is fundamentally ambiguous from a single equation; only the component of flow perpendicular to the local gradient is observable).

- **Lucas-Kanade (sparse, local).** Assumes flow is constant within a small window around each tracked point, turning the per-pixel underdetermined problem into an overdetermined least-squares problem solved via the normal equations:
```
[ΣIx²  ΣIxIy] [u]   [ΣIxIt]
[ΣIxIy ΣIy² ] [v] = -[ΣIyIt]
```
Used with a pyramidal (coarse-to-fine) scheme to handle larger displacements, and typically applied only at sparse, well-textured keypoints (e.g., Shi-Tomasi corners) since the window-constant-flow assumption breaks down on flat/textureless regions.
- **Horn-Schunck (dense, global).** Estimates flow at every pixel by minimizing brightness-constancy error plus a global smoothness penalty on the flow field's spatial gradient — denser but blurs motion boundaries and is more sensitive to large displacements than Lucas-Kanade.
- **Deep optical flow (FlowNet, RAFT).** Learn flow end-to-end from data instead of hand-derived constraints. RAFT builds a 4D all-pairs correlation volume between the two frames' feature maps and iteratively refines the flow estimate with a recurrent (GRU-based) update operator — current standard for accuracy on benchmarks like Sintel/KITTI.

```python
import cv2
import numpy as np

# Sparse (Lucas-Kanade) — track a set of keypoints across two frames
prev_pts = cv2.goodFeaturesToTrack(prev_gray, maxCorners=200, qualityLevel=0.01, minDistance=7)
next_pts, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, prev_pts, None)

# Dense (Farneback) — a per-pixel flow field
flow = cv2.calcOpticalFlowFarneback(prev_gray, next_gray, None,
                                     pyr_scale=0.5, levels=3, winsize=15,
                                     iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
# flow.shape == (H, W, 2)  -> flow[..., 0] = u, flow[..., 1] = v
```

**Action recognition (video classification):** extends image classification into the temporal dimension.
- **Two-Stream Networks (Simonyan & Zisserman, 2014):** one CNN stream processes RGB frames (appearance/spatial cues), a second CNN stream processes stacked optical-flow fields (motion/temporal cues); the two streams' predictions are fused (late fusion). Motion is made explicit via precomputed optical flow rather than learned implicitly.
- **3D CNNs (C3D, I3D):** replace 2D convolutions with **3D convolutions** that slide across height, width, *and* time jointly, learning spatiotemporal features directly from stacked RGB clips without needing precomputed optical flow. I3D "inflates" a pretrained 2D image classifier's 2D kernels into 3D by repeating them along the time axis, letting it bootstrap from ImageNet pretraining.
- **Video transformers (TimeSformer, ViViT):** extend ViT's patch-attention idea to space-time by tokenizing video into spatio-temporal patches and applying (often factorized, to control cost) attention across space and time separately, avoiding the full quadratic blowup of joint space-time attention.

**Pitfalls:**
- Optical flow's brightness-constancy assumption breaks down under large displacements, motion blur, lighting changes, and specular/transparent surfaces.
- The aperture problem means flow is fundamentally ill-posed on low-texture regions regardless of algorithm — always check flow confidence/consistency (e.g., forward-backward consistency check) before trusting it downstream.
- Occlusion/disocclusion: pixels visible in one frame but not the other have no true correspondence; most classical flow methods produce silently wrong (not flagged) estimates there.

---

### Face recognition and verification

**Pipeline:** `Detection → Alignment → Embedding → Matching`
1. **Detection:** locate face bounding boxes/landmarks in an image (MTCNN, RetinaFace, or a general-purpose detector fine-tuned for faces).
2. **Alignment:** use detected landmarks (eyes, nose, mouth corners) to warp the face crop to a canonical pose via a similarity/affine transform, removing in-plane rotation/scale variation before embedding.
3. **Embedding:** a CNN maps the aligned face crop to a fixed-length, typically **L2-normalized** vector (128-d for FaceNet, 512-d for ArcFace) such that embeddings of the same identity are close together and different identities are far apart under a simple distance metric.
4. **Matching:** compare embeddings via Euclidean distance or cosine similarity against a threshold.

**Verification (1:1)** — "is this the same person as this reference photo?" — a single distance-vs-threshold decision. **Recognition/identification (1:N)** — "who is this, out of N enrolled identities?" — compare a query embedding against a gallery and return the nearest match (or "no match" if all distances exceed a threshold).

**Triplet loss (FaceNet, Schroff et al. 2015):** for an anchor `a`, a positive `p` (same identity as `a`), and a negative `n` (different identity):
```
L = max(0, ||f(a) - f(p)||² - ||f(a) - f(n)||² + margin)
```
This directly optimizes the embedding geometry — pushing the anchor-positive distance to be smaller than the anchor-negative distance by at least `margin` — rather than optimizing a fixed softmax classification head. This is why new identities can be enrolled at inference time just by embedding a reference photo, with no retraining needed.

```python
import torch
import torch.nn.functional as F

def triplet_loss(anchor, positive, negative, margin=0.2):
    d_pos = (anchor - positive).pow(2).sum(dim=1)
    d_neg = (anchor - negative).pow(2).sum(dim=1)
    return F.relu(d_pos - d_neg + margin).mean()
```

**The triplet-mining problem:** most randomly sampled triplets already satisfy the margin (they're "easy," contributing near-zero gradient), so training requires actively mining **hard or semi-hard negatives** (a negative closer to the anchor than the positive, or just inside the margin) within each batch — a significant source of training complexity and instability.

**ArcFace (additive angular margin loss)** sidesteps explicit triplet mining by reformulating face embedding learning as a *classification*-style softmax loss over the angle between an embedding and each class's weight vector on a hypersphere, adding an angular margin `m` to the target class's angle before the softmax:
```
L = -log[ e^(s·cos(θ_yi + m)) / (e^(s·cos(θ_yi + m)) + Σ_{j≠yi} e^(s·cos θ_j)) ]
```
where `θ_yi` is the angle between the embedding and its true class's weight vector, `s` is a fixed scale, and `m` is the margin. Because every training sample is used directly (like ordinary classification) rather than needing to be assembled into informative triplets, ArcFace-style losses train more stably at scale (millions of identities) and are the current standard for production face-recognition embeddings.

**Pitfalls:**
- Verification threshold choice is a False-Accept-Rate (FAR) vs. False-Reject-Rate (FRR) tradeoff that must be tuned per deployment (e.g., unlocking a phone tolerates a different FAR/FRR balance than a bank KYC check) — always inspect the ROC/DET curve, not a single accuracy number.
- Presentation attacks (printed photo, video replay, mask) are **not** addressed by the embedding model itself — production systems need a separate **liveness detection** stage.
- Face recognition models are well-documented to show accuracy disparities across demographic groups when training data is not diverse — a core fairness/ethics concern (see Responsible AI/Fairness) rather than a purely architectural one.

---

### OCR pipeline: text detection + text recognition

Modern OCR is almost always factored into two stages (even when trained/shipped as a single pipeline):

1. **Text detection** — localize regions of the image containing text, as boxes or polygons:
   - **Regression-based (EAST):** directly regresses rotated bounding boxes/quadrilaterals per pixel/anchor, similar in spirit to anchor-based object detection.
   - **Segmentation-based (PSENet, DBNet):** predict a per-pixel text/non-text probability map (and often a shrunk "kernel" map to help separate adjacent text lines), then post-process the mask into polygons — handles curved/irregular/arbitrarily-oriented text much better than box regression.
   - **Anchor-based (CTPN):** adapts object-detection-style anchors to detect horizontal text lines as sequences of narrow proposals.
2. **Text recognition** — given a cropped text-line image, predict the character sequence:
   - **CRNN (CNN + RNN + CTC):** a CNN extracts a sequence of column features from the cropped image, a BiLSTM models sequential dependencies across characters, and a **CTC (Connectionist Temporal Classification)** loss trains the whole thing without needing pre-aligned per-character labels.
   - **Transformer-based (TrOCR):** a ViT-style image encoder feeds a Transformer text decoder, framing recognition as an image-to-text generation problem (analogous to image captioning).

**CTC loss, briefly:** the image-feature sequence and the target character sequence generally have different, variable lengths, and the true per-timestep alignment is unknown. CTC introduces a special **blank** token and defines the loss as the (negative log) sum over *all* possible per-timestep label paths that collapse — after merging repeated consecutive characters and removing blanks — to the target sequence. This makes it possible to train a sequence-to-sequence recognizer with only sequence-level (not per-character-aligned) labels.

**Classical baseline:** Tesseract (connected-component-based segmentation + trained character classifiers) — still a reasonable, dependency-light baseline for clean, scanned, horizontal documents; deep pipelines (PaddleOCR, TrOCR, cloud OCR APIs) dominate on real-world/scene text.

**Pitfalls:**
- Curved, rotated, or densely packed text breaks axis-aligned detectors — prefer polygon/segmentation-based detection for scene text (signs, packaging) vs. scanned documents.
- Recognition errors and detection errors compound in a two-stage pipeline but must be debugged separately — a wrong detection crop guarantees a wrong recognition regardless of recognizer quality.
- Script/language coverage is not free — multilingual, right-to-left, and vertical-text OCR generally need script-specific training data and sometimes different model heads.

---

### Human pose estimation

**Top-down approach:** run a person detector first, crop each detected person, then run a per-person keypoint-estimation network on each crop (e.g., simple baselines, HRNet). Accuracy tends to be higher (the keypoint network sees a normalized, zoomed-in single-person crop), but total inference cost scales with the number of people detected in the image.

**Bottom-up approach:** detect *all* keypoints across the whole image in one pass, then group them into individual person instances. **OpenPose** does this via **Part Affinity Fields (PAFs)** — learned 2D vector fields encoding the position and orientation of each limb — used to associate detected joint candidates with the correct person even in crowded, overlapping scenes. Runtime doesn't scale directly with the number of people, but the grouping/association step adds its own failure modes in dense crowds.

**Heatmap-based keypoint representation:** rather than directly regressing `(x, y)` coordinates for each keypoint (numerically ill-conditioned, and it forces a single network output to encode fine-grained spatial precision), most modern pose networks predict, per keypoint type, a spatial **heatmap** — a 2D Gaussian centered at the true keypoint location over the image/feature-map grid. The predicted location is the heatmap's arg-max (or a soft-argmax for sub-pixel precision). This reframes precise localization as a per-pixel density-estimation problem, which convolutional networks handle far more effectively than direct coordinate regression.

**Evaluation metrics:**
- **PCK (Percentage of Correct Keypoints):** a predicted keypoint is "correct" if within some distance threshold of the ground truth, normalized by a body-size measure (e.g., torso or head size) so the metric is scale-invariant across people/images.
- **OKS (Object Keypoint Similarity):** COCO's pose metric, conceptually analogous to IoU for boxes — a Gaussian-weighted similarity between predicted and ground-truth keypoints, normalized by object scale and per-keypoint visibility/difficulty constants; thresholding OKS at multiple levels and averaging gives an AP-style "pose mAP."

**3D pose estimation** (briefly): lifts 2D keypoints to 3D, or regresses 3D joint coordinates directly, for applications like AR/VR avatars, sports biomechanics, and human-robot interaction. Substantially harder than 2D pose from a single monocular image because depth is fundamentally ambiguous without additional cues (multi-view, depth sensors, or learned priors).

---

### Camera geometry and 3D vision: stereo, depth estimation, point clouds

**Pinhole camera model.** A 3D world point projects onto the 2D image plane via:
```
s · [u, v, 1]ᵗ = K · [R | t] · [X, Y, Z, 1]ᵗ
```
where `[R | t]` (the **extrinsics**) is the camera's rotation/translation relative to the world frame, and `K` (the **intrinsics**) encodes the camera's internal geometry:
```
K = [ fx  0   cx ]
    [ 0   fy  cy ]
    [ 0   0   1  ]
```
`fx, fy` are focal lengths in pixel units, `(cx, cy)` is the principal point. Intrinsics are typically estimated via **camera calibration** (e.g., Zhang's checkerboard method).

**Stereo vision and depth from disparity.** Two cameras with a known baseline `B` (distance between camera centers) view the same scene; after **rectification** (warping both images so corresponding points lie on the same horizontal scanline, reducing correspondence search to 1D), a 3D point's horizontal image position differs between the two views by the **disparity** `d`. Depth relates to disparity as:
```
Z = f · B / d
```
Larger disparity means closer objects; disparity approaches 0 as depth approaches infinity. Finding `d` for every pixel is the **stereo correspondence problem**, solved classically via block matching (e.g., semi-global matching) or with deep stereo-matching networks that build a cost volume over candidate disparities and regress/argmax it.

**Monocular depth estimation.** Estimating per-pixel depth from a *single* RGB image is fundamentally ill-posed (one 2D image is consistent with infinitely many 3D scenes), so models must exploit learned monocular cues (relative object size, texture gradient, occlusion boundaries, scene priors) from data.
- **Supervised:** train directly against ground-truth depth from LiDAR or stereo.
- **Self-supervised (Monodepth/Monodepth2-style):** train on stereo pairs or monocular video with no depth labels at all — using a **photometric/view-synthesis loss**: predict depth (and, for monocular video, relative pose), warp one view into another using the predicted depth, and penalize the pixel-wise reconstruction error between the warped and real image.

**Point clouds and PointNet (concept).** A point cloud is an unordered set of 3D points (from LiDAR, stereo, or depth cameras), each possibly carrying extra features (color, intensity, normal). Because point order is arbitrary, a model must be **permutation-invariant**. PointNet achieves this by applying a shared per-point MLP (the same transformation to every point independently) followed by a symmetric aggregation (global max pooling) across all points — guaranteeing the output doesn't depend on input point order, while still learning directly from raw, unordered 3D points rather than requiring voxelization or projection onto a 2D grid first.

**Applications:** autonomous driving (LiDAR point clouds fused with camera images for 3D object detection), robotic grasping/manipulation, AR/VR scene reconstruction, and SLAM/structure-from-motion (which builds on the classical keypoint matching — SIFT/ORB — already covered under Image Fundamentals).

**Pitfalls:**
- Confusing intrinsics (fixed per-camera, from calibration) with extrinsics (change every time the camera moves) — mixing them up is a common source of projection bugs.
- Stereo depth degrades badly on textureless or repetitive surfaces (correspondence search becomes ambiguous), and disparity-to-depth is highly sensitive to small disparity errors at long range (`Z = fB/d` — the derivative of `Z` w.r.t. `d` blows up as `d → 0`, so far-away points have very noisy depth estimates).
- Monocular depth models can produce plausible-looking but metrically wrong (scale-ambiguous) depth unless explicitly trained/calibrated for absolute scale — self-supervised monocular-video methods in particular often only recover depth up to an unknown, possibly per-sequence, scale factor.

**Cross-file note:** this section stays at conceptual/interview depth for 3D vision and video; full SLAM/robotics system design lives outside this syllabus's scope, and deep generative video/3D models (NeRF-style, video diffusion) belong to Generative AI/LLMs (file 11) rather than here.

### Interview Questions

1. **Q: Derive the optical flow constraint equation and explain what it does and doesn't tell you about the true motion at a pixel.**
   A: Starting from the brightness constancy assumption `I(x,y,t) = I(x+u·dt, y+v·dt, t+dt)`, a first-order Taylor expansion of the right side and cancellation of `I(x,y,t)` gives `Ix·u + Iy·v + It ≈ 0` after dividing by `dt`. This single linear equation constrains only the component of flow along the local intensity gradient direction; the component perpendicular to the gradient (along an edge/uniform region) is unconstrained — the aperture problem. This is why single-pixel flow is underdetermined and methods like Lucas-Kanade must add an assumption (e.g., local constancy over a window) to get enough equations to solve for both `u` and `v`.

2. **Q: Contrast Lucas-Kanade and Horn-Schunck optical flow, and explain why deep learning methods like RAFT have largely superseded both for accuracy-critical applications.**
   A: Lucas-Kanade assumes flow is locally constant within a small window, giving a sparse, fast, per-point least-squares solution, typically applied at well-textured keypoints. Horn-Schunck instead estimates a dense flow field over the whole image by adding a global smoothness regularizer, but tends to over-smooth motion boundaries. Both rely on the brightness-constancy assumption and struggle with large displacements, occlusion, and textureless regions. RAFT instead learns flow end-to-end, builds an explicit all-pairs correlation volume between frames, and iteratively refines the estimate with a recurrent update — learned features and iterative refinement handle large motions, textureless regions, and lighting changes far more robustly than hand-derived constraints.

3. **Q: How do two-stream networks and 3D CNNs differ in how they incorporate motion information for action recognition?**
   A: Two-stream networks make motion explicit: a separate CNN stream processes precomputed optical-flow fields (motion) alongside a stream processing raw RGB frames (appearance), fusing both streams' predictions. 3D CNNs (e.g., I3D) instead apply convolution jointly across height, width, and time on stacked RGB clips, learning to extract spatiotemporal motion cues implicitly from the data rather than relying on a separate, precomputed optical-flow input — trading the cost/latency of computing optical flow for the cost of 3D convolutions and needing enough data/pretraining (e.g., via kernel inflation from a 2D backbone) to learn good temporal features.

4. **Q: Walk through the face verification pipeline end to end, and explain why alignment is a separate step from detection.**
   A: Detection locates the face and (usually) a small set of landmarks (eyes, nose, mouth). Alignment then uses those landmarks to warp the face crop to a canonical pose via a similarity/affine transform, removing in-plane rotation and scale variation before the embedding network ever sees the image — without this normalization, the embedding network would have to learn rotation/scale invariance itself, wasting capacity and hurting accuracy. The aligned crop is passed through an embedding CNN to get a fixed-length, L2-normalized vector, and verification is simply thresholding the distance (Euclidean or cosine) between two such vectors.

5. **Q: Why does triplet loss let you enroll a brand-new identity at inference time with no retraining, whereas a standard softmax classifier can't?**
   A: A softmax classifier's output layer has one fixed weight vector per known class, so adding a new identity requires adding (and training) a new output unit — retraining is required. Triplet loss instead directly shapes the embedding space so that same-identity images are close and different-identity images are far apart, with no notion of a fixed class set at all; verifying or enrolling a new identity is just computing its embedding and comparing/storing distances, which requires no architectural change or retraining.

6. **Q: What is triplet mining, and why is it necessary for training FaceNet-style embeddings effectively?**
   A: Most randomly sampled anchor-positive-negative triplets already satisfy the margin condition (the negative is already far enough from the anchor), contributing near-zero gradient and wasting training compute. Triplet mining selects hard or semi-hard negatives — negatives that are closer to the anchor than the positive, or just inside the margin — within each batch, so gradient updates come from triplets the model is actually getting wrong or nearly wrong, which is essential for the loss to make meaningful progress but adds real engineering complexity (mining strategy, batch construction).

7. **Q: How does ArcFace's angular margin loss avoid the triplet-mining problem entirely?**
   A: ArcFace reformulates embedding learning as an ordinary classification softmax loss over the angle between an embedding and each identity's learned weight vector on a hypersphere, adding a margin directly to the true class's angle before the softmax. Because this uses every training sample individually (exactly like standard classification), it never needs to construct or mine informative triplets/pairs, making training far more stable and scalable to millions of identities — at the cost of needing a fixed set of identity classes during training (though the resulting embeddings still generalize to unseen identities at inference).

8. **Q: Describe the two stages of a typical OCR pipeline and one modern approach for each.**
   A: (1) Text detection localizes regions containing text — a segmentation-based method like DBNet predicts a per-pixel text probability map and post-processes it into polygons, handling curved/rotated text better than simple box regression. (2) Text recognition reads the character sequence from each cropped region — CRNN extracts a CNN feature sequence, models it with a BiLSTM, and trains with CTC loss to avoid needing per-character-aligned labels; alternatively, transformer-based models like TrOCR treat recognition as image-to-text generation.

9. **Q: Why is CTC loss necessary for training a sequence recognizer like CRNN, rather than just using per-timestep cross-entropy?**
   A: The image-feature sequence's length and the target character sequence's length generally differ, and the true alignment between them (which feature timestep corresponds to which character) is unknown and would be expensive to annotate. CTC introduces a blank token and sums the probability over every possible timestep-to-label path that collapses (after removing blanks and merging repeated characters) to the correct target sequence, allowing the network to be trained end-to-end from sequence-level labels alone, without any explicit alignment supervision.

10. **Q: Compare top-down and bottom-up human pose estimation, and explain the tradeoff in a crowded scene.**
    A: Top-down first detects each person, then runs keypoint estimation on each cropped, normalized region — generally more accurate per person but its cost scales linearly with the number of people detected, which can become slow in a crowd. Bottom-up detects all keypoints across the image in one pass, then groups them into people (e.g., via OpenPose's Part Affinity Fields) — cost doesn't scale with the number of people, but the grouping/association step becomes more error-prone as people overlap heavily, since PAFs/keypoint proximity cues get ambiguous in dense crowds.

11. **Q: Why do modern pose estimation networks predict heatmaps rather than directly regressing (x, y) keypoint coordinates?**
    A: Direct coordinate regression asks a network to output a single precise number per keypoint from a global pooled representation, which is numerically ill-conditioned and discards useful spatial information that convolutional feature maps naturally encode. Predicting a per-keypoint heatmap (a 2D Gaussian centered at the true location) turns localization into a per-pixel density-estimation problem, which CNNs are much better suited to, and it naturally provides a confidence/uncertainty signal (the heatmap's peakedness) that a single coordinate regression does not.

12. **Q: Explain how depth is recovered from a stereo camera pair, and why depth estimates for distant objects are especially noisy.**
    A: After rectifying both images so corresponding points share the same row, the stereo correspondence problem finds, for each pixel, the horizontal offset (disparity `d`) to its match in the other image; depth follows from `Z = f·B/d` given focal length `f` and baseline `B`. Because `Z` is inversely proportional to `d`, a fixed small error in disparity translates into a depth error that grows quadratically as `d` shrinks (i.e., as objects get farther away) — `dZ/dd = -fB/d²` — so distant points, which naturally have small disparities, are far more sensitive to correspondence noise than nearby points.

13. **Q: Why does PointNet apply a shared per-point MLP followed by a symmetric pooling function instead of, say, a standard CNN or RNN over the raw point list?**
    A: A point cloud has no inherent ordering, so any model consuming it directly must be invariant to the order in which points are listed. A CNN or RNN over the raw point sequence would produce different outputs for different (arbitrary) orderings of the same physical point cloud, which is incorrect. PointNet's shared per-point MLP treats every point identically and independently, and the final symmetric aggregation (e.g., max pooling across points) is provably permutation-invariant regardless of input order, while still letting the network learn directly from unordered 3D coordinates without first voxelizing or projecting onto a 2D grid (which would lose precision or waste computation on empty space).

14. **Q: A self-supervised monocular depth model trained on dashcam video produces visually plausible depth maps, but the absolute distances are wrong by a consistent multiplicative factor across the whole video. What's the likely cause?**
    A: Self-supervised monocular-video depth training (à la Monodepth2) optimizes a photometric/view-synthesis loss between warped and real frames, which is invariant to a global scale ambiguity — the model can jointly scale its predicted depth and predicted camera translation by any constant factor and still perfectly reconstruct the photometric loss, since only their product matters for the warp. Without an external metric scale reference (e.g., known camera height, stereo baseline, or LiDAR supervision at some point), the model has no signal to pin down absolute scale, only relative/scaled depth — this is expected behavior, not a bug, and is typically resolved by calibrating against a known reference distance in the deployment scene.

15. **Q: You need to add face verification to a mobile banking app's KYC (know-your-customer) flow. What would you consider beyond simply picking a high-accuracy embedding model?**
    A: Liveness detection to defend against presentation attacks (printed photo, screen replay, mask) since the embedding model alone cannot distinguish a live face from a well-presented static image; careful threshold tuning on the FAR/FRR tradeoff appropriate for financial KYC (favoring low FAR — false accepts are a fraud/compliance risk — even at some FRR cost to genuine users); testing accuracy across demographic subgroups given documented disparities in face recognition systems trained on non-diverse data; on-device latency/model-size constraints for the embedding network; and a clear fallback path (e.g., manual review) for verification failures rather than a hard binary accept/reject.

---

## Practical CV

### Data augmentation techniques

**Why augmentation helps generalization:** deep vision models have very high capacity and can easily overfit to spurious/incidental characteristics of the finite training set (specific lighting, exact object position, background clutter). Augmentation synthetically expands the effective training distribution by applying label-preserving transformations, teaching the model invariances it should have (e.g., a cat is still a cat if flipped horizontally or slightly rotated), which directly improves robustness and validation/test generalization.

| Technique | What it does | Typical use |
|---|---|---|
| Horizontal/vertical flip | Mirrors the image | Almost always safe for horizontal flip on natural images; vertical flip risky (upside-down objects rare); NOT safe if left/right or up/down orientation is semantically meaningful (e.g., text, medical laterality) |
| Random rotation | Rotates by a random angle | Useful when orientation varies naturally (aerial imagery); risky for orientation-sensitive tasks |
| Color jitter | Randomly perturbs brightness/contrast/saturation/hue | Improves robustness to lighting/camera variation |
| Random crop/resize | Crops a random sub-region then resizes | Simulates scale/translation variation, very standard (used in ImageNet training pipelines) |
| Cutout | Masks out a random rectangular region (set to 0/mean) | Forces model to not over-rely on any single localized feature/region |
| Mixup | Blends two images and their labels: `x = λx_1 + (1-λ)x_2`, `y = λy_1 + (1-λ)y_2` | Smooths decision boundaries, improves calibration and robustness |
| CutMix | Cuts a patch from one image and pastes it into another; label is mixed proportionally to pasted area | Combines Cutout's regional occlusion robustness with Mixup's label smoothing; often outperforms both individually |

**Mixup formula:**
```
x_mix = λ·x_i + (1-λ)·x_j
y_mix = λ·y_i + (1-λ)·y_j          (soft label combination)
λ ~ Beta(α, α)                      (α typically 0.2–0.4)
```

**CutMix formula:** samples a bounding box region, replaces that region in image A with the same region from image B; the mixed label weight `λ` = fraction of image area taken from image B.

```python
import torch
import numpy as np

def mixup(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0))
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam
    # loss = lam * criterion(pred, y) + (1-lam) * criterion(pred, y[idx])

def cutmix(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0))
    H, W = x.size(2), x.size(3)
    cut_rat = np.sqrt(1. - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1, x2 = np.clip(cx - cut_w // 2, 0, W), np.clip(cx + cut_w // 2, 0, W)
    y1, y2 = np.clip(cy - cut_h // 2, 0, H), np.clip(cy + cut_h // 2, 0, H)
    x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam_adjusted = 1 - ((x2 - x1) * (y2 - y1) / (W * H))
    return x, y, y[idx], lam_adjusted
```

**Detection/segmentation-specific augmentation caveat:** bounding boxes/masks must be transformed *consistently* with the image — e.g., a horizontal flip must also flip box x-coordinates (`x_new = W - x_old`), and a crop must clip/discard boxes/masks that fall outside the crop region. Libraries like `albumentations` handle this synchronization automatically and are strongly preferred over manual implementations for detection/segmentation augmentation.

```python
import albumentations as A
transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.3),
    A.RandomResizedCrop(height=512, width=512, scale=(0.8, 1.0), p=0.5),
], bbox_params=A.BboxParams(format="coco", label_fields=["class_labels"]))
```

**Pitfalls:**
- Applying augmentations that break label validity — e.g., flipping digit "6" to look like "9," rotating text, or vertically flipping images where "up" is semantically meaningful.
- Forgetting to transform bounding boxes/masks alongside the image in detection/segmentation pipelines — this silently corrupts training labels.
- Over-augmenting: excessively aggressive augmentation can make the task too hard/unrealistic and hurt convergence rather than help generalization.

---

### Handling class imbalance in detection/segmentation

Object detection and segmentation are *inherently* imbalanced problems: most of any image is background, and object-class frequency in real datasets is often long-tailed (e.g., "car" appears far more often than "unicycle").

**Techniques:**

| Technique | Mechanism | Where used |
|---|---|---|
| **Focal Loss** | Down-weights well-classified ("easy") examples so the loss focuses on hard/misclassified examples: `FL(p) = -α(1-p)^γ·log(p)` | One-stage detectors (RetinaNet), addresses foreground/background imbalance |
| **Hard negative mining** | Explicitly selects the highest-loss negative examples for training rather than using all/random negatives | Two-stage detectors, SSD |
| **Class-balanced / weighted loss** | Weight each class's loss contribution inversely proportional to its frequency | Classification, segmentation |
| **Oversampling / resampling** | Oversample images/patches containing rare classes; undersample dominant background-only images | Any detection/segmentation dataset with skewed class distribution |
| **Dice / Tversky Loss** | Directly optimizes overlap-based metrics that are less sensitive to raw pixel-count imbalance | Segmentation, especially medical imaging |
| **Copy-paste augmentation** | Synthetically composites rare-class object instances into new backgrounds | Detection/segmentation with rare object classes |

**Focal Loss derivation intuition:** standard cross-entropy `CE(p) = -log(p)` gives significant loss even for confidently-correct easy examples (small but nonzero, and with a huge number of easy examples, their summed loss can still dominate training signal). Focal loss multiplies by `(1-p)^γ`, which shrinks toward 0 rapidly as `p` (the predicted probability for the correct class) approaches 1 — so well-classified examples contribute vanishingly little loss, letting gradient updates focus on the harder, still-misclassified examples. `α` additionally reweights the relative importance of foreground vs. background classes.

```python
import torch
import torch.nn.functional as F

def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    ce_loss = F.cross_entropy(logits, targets, reduction="none")
    p = torch.exp(-ce_loss)                 # p = predicted probability of true class
    focal_term = (1 - p) ** gamma
    return (alpha * focal_term * ce_loss).mean()
```

**Pitfalls:**
- Blindly applying class weights computed from raw class frequency without considering per-image co-occurrence structure (detection/segmentation labels aren't i.i.d. like simple classification labels).
- Over-oversampling rare classes to the point of severe overfitting on a handful of unique rare-class instances.
- Using only accuracy/pixel-accuracy to monitor training progress on imbalanced tasks — always track per-class metrics (per-class AP, per-class IoU) to catch silent rare-class failures.

---

### Model evaluation and deployment considerations: latency, model size, quantization/pruning

**Key deployment metrics:**

| Metric | What it measures | Why it matters |
|---|---|---|
| Latency | Time per inference (ms) | Real-time constraints (autonomous driving, video processing) |
| Throughput | Inferences per second (often batched) | Server-side/batch processing cost efficiency |
| Model size | Disk/memory footprint (MB) | Mobile/edge device storage and RAM constraints |
| FLOPs / MACs | Theoretical compute cost | Proxy for latency/energy use, hardware-independent comparison |
| Power/energy consumption | Joules per inference | Battery-powered/embedded deployment |

**Quantization.** Reduces numerical precision of weights/activations (typically FP32 → INT8, sometimes FP16 or even INT4/binary), shrinking model size (~4x for FP32→INT8) and often significantly speeding up inference on hardware with optimized low-precision kernels, at some accuracy cost.

| Type | Description |
|---|---|
| Post-Training Quantization (PTQ) | Quantize an already-trained FP32 model directly, optionally using a small calibration dataset to determine optimal quantization ranges; fast/simple but can lose more accuracy |
| Quantization-Aware Training (QAT) | Simulate quantization effects (fake quantize ops) during training/fine-tuning so the model learns to be robust to reduced precision; more effort, typically better accuracy retention than PTQ |
| Dynamic quantization | Quantize weights ahead of time, activations computed/quantized dynamically at runtime | 
| Static quantization | Both weights and activations quantized ahead of time using calibration data |

```python
import torch

# Post-training dynamic quantization example (PyTorch)
model_fp32 = MyModel()
model_fp32.eval()
model_int8 = torch.quantization.quantize_dynamic(
    model_fp32, {torch.nn.Linear}, dtype=torch.qint8
)
```

**Pruning.** Removes redundant/low-importance weights or entire structures (channels, filters, layers) from a trained network.

| Type | Description | Hardware benefit |
|---|---|---|
| Unstructured (weight) pruning | Zero out individual low-magnitude weights | Reduces parameter count/storage, but needs sparse-matrix hardware/kernel support to realize speedup |
| Structured pruning | Remove entire channels/filters/layers | Directly reduces dense compute (FLOPs) — realizes actual speedup on standard hardware without special sparse kernels |

**Knowledge distillation** (related technique): train a smaller "student" model to mimic a larger "teacher" model's output distribution (soft labels/logits), often recovering much of the teacher's accuracy in a far more compact/fast model — commonly combined with quantization/pruning in a full compression pipeline.

**Deployment format/tooling considerations:**
- **ONNX**: framework-agnostic model interchange format, widely supported for cross-platform deployment (PyTorch/TensorFlow → ONNX → various runtimes).
- **TensorRT** (NVIDIA): highly optimized inference engine for NVIDIA GPUs, applies layer fusion, precision calibration (FP16/INT8), and kernel auto-tuning.
- **TFLite / Core ML**: mobile-optimized runtimes (Android/iOS respectively) supporting quantized models for on-device inference.
- **Batching strategy**: larger batches improve GPU throughput but increase per-request latency — real-time applications often need batch size 1 (or small dynamic batching windows) to bound latency.

**Practical accuracy-vs-efficiency tradeoff workflow:**
1. Establish an accuracy baseline with a full-precision model.
2. Apply structured pruning and/or choose an efficient architecture (MobileNet/EfficientNet-lite) suited to the target hardware.
3. Apply quantization (prefer QAT if accuracy loss from PTQ is unacceptable).
4. Benchmark actual on-device latency (not just FLOPs, which don't always correlate perfectly with real-world latency due to memory bandwidth, hardware-specific kernel support, etc.).
5. Validate accuracy on a held-out set after each compression step — compression can introduce systematic biases (e.g., disproportionately hurting rare classes) that a single aggregate accuracy number can mask.

**Pitfalls:**
- Assuming lower FLOPs always means lower latency — memory bandwidth, cache behavior, and hardware-specific kernel support (e.g., depthwise convs are FLOP-cheap but not always latency-cheap on all hardware) can decouple the two.
- Quantizing without checking accuracy on a representative validation set, especially per-class/per-region — quantization error can disproportionately affect edge cases or rare classes.
- Ignoring preprocessing/postprocessing latency (image decode/resize, NMS) when benchmarking end-to-end deployment latency — the neural network forward pass is often not the only bottleneck.

### Interview Questions

1. **Q: Why does data augmentation improve generalization instead of just adding noise?**
   A: Augmentation applies label-preserving transformations that reflect real-world variation the model will encounter at test time (different lighting, viewpoint, scale, occlusion). This teaches the model invariances it should have, effectively expanding the training distribution to better approximate the true underlying data distribution, reducing overfitting to incidental characteristics of the specific finite training set.

2. **Q: Explain Mixup and why blending both images and labels (rather than just images) is important.**
   A: Mixup creates a synthetic training example by linearly interpolating both the pixel values (`x = λx_1+(1-λ)x_2`) and the labels (`y = λy_1+(1-λ)y_2`) of two random training samples. Blending only the images while keeping a single hard label would give an incorrect/misleading target since the blended image is genuinely ambiguous between the two classes — blending the label proportionally teaches the model calibrated, smoother decision boundaries and improves generalization/robustness to label noise.

3. **Q: When is horizontal flip augmentation inappropriate?**
   A: When left-right orientation is semantically meaningful to the task — e.g., text/digit recognition (flipped digits/letters can become invalid or a different character), medical imaging where organ laterality matters (e.g., distinguishing left vs. right lung), or traffic sign recognition where directional arrows have fixed real-world meaning.

4. **Q: Derive/explain the intuition behind Focal Loss and why it helps with class imbalance in detection.**
   A: Standard cross-entropy loss `-log(p)` still contributes meaningfully to total loss even for easy, well-classified examples, and since background/easy negatives vastly outnumber hard positives in detection, their aggregate loss can dominate training and drown out useful gradient signal from hard examples. Focal loss multiplies cross-entropy by a modulating factor `(1-p)^γ` that shrinks rapidly toward zero as the predicted probability for the correct class approaches 1, effectively down-weighting easy examples' contribution and letting the loss focus on hard, misclassified examples — directly addressing the foreground/background imbalance without needing explicit resampling.

5. **Q: What must you be careful about when applying augmentation to an object detection dataset (versus plain classification)?**
   A: Any geometric transformation applied to the image (flip, crop, rotate, scale) must be applied consistently to the corresponding bounding box coordinates (and/or segmentation masks) — e.g., a horizontal flip requires `x_new = image_width - x_old` for box coordinates, and cropping requires clipping or discarding boxes that fall outside the crop region. Failing to synchronize these transformations silently corrupts the training labels.

6. **Q: Compare quantization and pruning as model compression techniques.**
   A: Quantization reduces the numerical precision of weights/activations (e.g., FP32→INT8), shrinking model size and often speeding up inference via low-precision hardware kernels, without changing model architecture/structure. Pruning removes redundant weights or structures (channels/filters/layers) entirely, reducing parameter count and (if structured) FLOPs directly. They are complementary and often combined in the same compression pipeline (prune, then quantize, then possibly fine-tune/distill).

7. **Q: What's the difference between post-training quantization (PTQ) and quantization-aware training (QAT), and when would you choose one over the other?**
   A: PTQ quantizes an already-trained full-precision model directly (optionally using a small calibration set), requiring no retraining — fast and simple, but can suffer larger accuracy drops, especially for architectures sensitive to precision loss. QAT simulates quantization effects during training/fine-tuning (via fake-quantize operations), letting the model adapt its weights to be robust to reduced precision — more effort/compute but typically much better accuracy retention. Choose QAT when PTQ's accuracy drop is unacceptable and you have the compute/time budget to retrain.

8. **Q: Why might structured pruning provide more realized speedup than unstructured pruning, even at the same "percentage of weights removed"?**
   A: Unstructured pruning zeroes out individual weights scattered throughout dense weight matrices, producing sparse matrices that standard dense-matrix hardware/kernels can't exploit for speedup without specialized sparse computation support — so FLOPs "on paper" go down but wall-clock latency often doesn't improve much. Structured pruning removes entire channels/filters/layers, directly shrinking the dense matrix dimensions, which standard hardware and existing dense kernels can immediately exploit for real speedup.

9. **Q: You quantize a detection model to INT8 and overall mAP only drops slightly, but you notice the rare "pedestrian at night" class's recall drops significantly. What would you investigate/do?**
   A: This suggests quantization error disproportionately affects a specific data subset/edge case not well represented in the (possibly unrepresentative) calibration set used for PTQ — investigate per-class/per-condition metrics (not just aggregate mAP) after quantization; consider using a calibration set that better represents this subgroup; consider QAT (which lets the model adapt weights to reduce precision-induced errors) instead of PTQ; or selectively keep sensitive layers/heads at higher precision (mixed-precision quantization).

10. **Q: What is knowledge distillation, and how can it be combined with quantization/pruning in a deployment pipeline?**
    A: Knowledge distillation trains a smaller "student" model to mimic a larger "teacher" model's output distribution (often using soft/softened logits as targets, which carry more information than hard labels), recovering much of the teacher's accuracy in a more compact model. In a full compression pipeline, distillation is often applied first (train a smaller, efficient architecture using the full teacher's guidance), and the resulting distilled model is then further pruned and/or quantized, since starting from a strong architecture with distilled knowledge tends to tolerate subsequent compression better than compressing directly from scratch.

11. **Q: Why is measured on-device latency often preferred over theoretical FLOPs when comparing models for deployment?**
    A: FLOPs are a hardware-independent theoretical measure of compute, but real latency also depends on memory bandwidth/access patterns, cache behavior, degree of hardware-specific kernel optimization/parallelism support (e.g., depthwise separable convolutions are FLOP-efficient but not always well-optimized/fast on every hardware/runtime), and fixed overheads (data transfer, preprocessing) — two models with similar FLOPs can have meaningfully different real-world latency on the same target hardware.

12. **Q: In a highly imbalanced segmentation task, why might combining cross-entropy loss with Dice loss work better than either alone?**
    A: Cross-entropy provides stable, well-behaved gradients especially early in training and handles per-pixel classification generally well, but is dominated by the majority class under severe imbalance. Dice loss directly targets overlap quality and is much more robust to imbalance (a small foreground region can still produce a strong gradient signal), but can be less numerically stable early in training (e.g., when predictions are near-random/near-zero overlap) and doesn't explicitly calibrate per-pixel probabilities as well as cross-entropy. A weighted combination gets stable training dynamics from cross-entropy plus imbalance robustness and boundary quality from Dice.

13. **Q: How would you decide the batch size to use at inference time for a real-time video analytics application versus a batch-processing offline pipeline?**
    A: For real-time applications, prioritize low per-request latency — use small batch sizes (often batch size 1, or a small dynamic batching window with a tight timeout) to minimize the delay any single frame waits before being processed. For offline batch processing (no strict latency deadline per item), use larger batch sizes to maximize GPU throughput/utilization and minimize total processing time/cost across the whole dataset, since latency per individual item is not the binding constraint.

14. **Q: What's the risk of using CutMix or Mixup in a task where object localization matters (e.g., as a pretraining step before fine-tuning for detection)?**
    A: These augmentations create composite images with mixed/soft labels that don't have a single well-defined bounding box or precise spatial ground truth for the mixed regions, so they are generally applied to whole-image classification pretraining rather than directly to detection/segmentation training with per-object localization labels — using them naively for detection without adapting the label mixing to per-object structure can create inconsistent or ill-defined localization targets.

15. **Q: A production vision model works well in validation but degrades in the field. Beyond accuracy metrics, what deployment-specific factors would you investigate?**
    A: Check for train/serve preprocessing mismatches (resize/normalization/color-space differences between training pipeline and production inference code), distribution shift between validation data and real-world deployment conditions (lighting, camera hardware, resolution, compression artifacts from video encoding), quantization/compression-induced accuracy degradation not caught by aggregate validation metrics, latency-driven corner-cutting (e.g., excessive downsampling or frame-skipping in a real-time pipeline), and whether the validation set itself was representative of true field conditions (e.g., collected under different camera/lighting setups than deployment).

---

## Rapid-Fire Interview Q&A

| # | Question | Answer |
|---|---|---|
| 1 | What does IoU stand for and range from? | Intersection over Union; ranges from 0 (no overlap) to 1 (perfect overlap) |
| 2 | What's the main purpose of pooling layers in a CNN? | Downsample spatial resolution, reduce computation, add mild translation invariance |
| 3 | What does a 1x1 convolution primarily do? | Mixes/projects channel information without changing spatial dimensions |
| 4 | What is the key innovation of ResNet? | Residual/skip connections that let gradients flow directly, enabling very deep network training |
| 5 | What does NMS stand for and do? | Non-Max Suppression; removes duplicate overlapping detection boxes, keeping the highest-confidence one |
| 6 | Two-stage vs one-stage detector: which is generally faster? | One-stage (e.g., YOLO, SSD) |
| 7 | What does mAP stand for? | Mean Average Precision |
| 8 | What is the Dice coefficient primarily used to evaluate? | Segmentation overlap quality, especially in medical imaging |
| 9 | What architecture introduced skip connections between encoder and decoder for segmentation? | U-Net |
| 10 | What does Mask R-CNN add on top of Faster R-CNN? | A per-RoI mask prediction branch (small FCN) for instance segmentation |
| 11 | What replaces convolution's local receptive field in a Vision Transformer? | Global self-attention across all patch tokens |
| 12 | How is an image converted into a sequence for ViT? | Split into fixed-size patches, flatten, linearly project (patch embedding) |
| 13 | Why does ViT need positional embeddings? | Self-attention is permutation-invariant; positional embeddings inject spatial order info |
| 14 | Name two contrastive self-supervised vision pretraining methods. | SimCLR, MoCo |
| 15 | What does MAE mask and reconstruct? | Randomly masks image patches; reconstructs their raw pixel values from visible patches |
| 16 | What is CLIP's training objective? | Contrastive alignment of image and text embeddings from (image, caption) pairs |
| 17 | How does CLIP enable zero-shot classification? | Compare image embedding similarity to text embeddings of class-describing prompts |
| 18 | What does MobileNet use to reduce compute? | Depthwise separable convolutions |
| 19 | What does EfficientNet's compound scaling jointly scale? | Network depth, width, and input resolution |
| 20 | What's the formula for Dice in terms of IoU? | Dice = 2·IoU / (1 + IoU) |
| 21 | What loss handles foreground/background imbalance in one-stage detectors? | Focal Loss |
| 22 | What's the difference between semantic and instance segmentation? | Semantic labels pixels by class only; instance also separates individual object instances |
| 23 | What color space separates hue from intensity, useful for lighting-robust color thresholding? | HSV |
| 24 | What edge detector uses non-max suppression and hysteresis thresholding? | Canny |
| 25 | What's the main tradeoff of quantizing a model to INT8? | Reduced size/faster inference vs. potential accuracy loss |
| 26 | What's the difference between structured and unstructured pruning re: speedup? | Structured pruning gives real speedup on standard hardware; unstructured needs sparse kernel support |
| 27 | What does RoI Align fix compared to RoI Pooling? | Removes coordinate quantization, using bilinear interpolation for precise feature alignment |
| 28 | What is the receptive field of a neuron? | The region of the original input image that affects that neuron's output |
| 29 | What augmentation blends two images and their labels proportionally? | Mixup |
| 30 | What augmentation pastes a patch from one image onto another? | CutMix |
| 31 | What's the key difference between anchor-based and anchor-free detectors? | Anchor-based uses predefined reference boxes for regression targets; anchor-free predicts directly from points/pixels (e.g., object centers) |
| 32 | What does GAP stand for and replace? | Global Average Pooling; replaces Flatten+large FC layer before classification head |
| 33 | What is transfer learning's core assumption? | Features learned on a large source dataset are useful/transferable to a related target task |
| 34 | Name the classical descriptor known for rotation/scale invariance using Difference-of-Gaussians. | SIFT |
| 35 | What does HOG capture, and what classical detector used it? | Gradient orientation histograms capturing shape/edges; used in pedestrian detection with linear SVM |
| 36 | What's the fast, binary-descriptor alternative to SIFT often used in SLAM? | ORB |
| 37 | What does panoptic segmentation unify? | Semantic segmentation of "stuff" + instance segmentation of "things," covering every pixel once |
| 38 | Why is pixel accuracy a poor segmentation metric under class imbalance? | Dominated by majority class; can be high even with poor minority-class segmentation |
| 39 | What's the key idea behind knowledge distillation? | Train a smaller student model to mimic a larger teacher model's output distribution |
| 40 | What does the Beta distribution parameter α control in Mixup? | The shape of the mixing coefficient λ's distribution (how aggressive the mixing is) |
| 41 | What assumption does optical flow rely on? | Brightness constancy — a point's intensity stays the same as it moves between frames |
| 42 | What is the "aperture problem" in optical flow? | Motion along a locally uniform edge/texture is ambiguous from a single local constraint — only the gradient-direction component of flow is observable |
| 43 | What's the difference between Lucas-Kanade and Horn-Schunck optical flow? | Lucas-Kanade: sparse, local constant-flow assumption, least-squares; Horn-Schunck: dense, global smoothness regularization |
| 44 | What does RAFT use to estimate optical flow? | An all-pairs correlation volume between frame features, iteratively refined by a recurrent (GRU-based) update |
| 45 | What loss function underlies FaceNet-style face embeddings? | Triplet loss |
| 46 | What problem does ArcFace's angular margin loss avoid that triplet loss suffers from? | The need for explicit hard/semi-hard triplet mining |
| 47 | What are the two stages of a typical OCR pipeline? | Text detection (localize text regions) and text recognition (read the character sequence) |
| 48 | What loss allows training a sequence recognizer (e.g., CRNN) without per-character-aligned labels? | CTC (Connectionist Temporal Classification) loss |
| 49 | What's the difference between top-down and bottom-up human pose estimation? | Top-down detects people first then estimates keypoints per crop; bottom-up detects all keypoints then groups them into people |
| 50 | Why do modern pose networks predict heatmaps instead of regressing (x, y) coordinates directly? | Heatmap prediction turns localization into a per-pixel density-estimation task, which CNNs handle far better than direct coordinate regression |
| 51 | What's the formula relating stereo disparity to depth? | Z = f·B / d (focal length × baseline / disparity) |
| 52 | What property must a model consuming a point cloud satisfy, and how does PointNet achieve it? | Permutation invariance; via a shared per-point MLP followed by a symmetric aggregation (e.g., max pooling) |
| 53 | What loss lets a monocular depth model train without any ground-truth depth labels? | A photometric/view-synthesis reconstruction loss between a warped view and the real target view |

---

*End of Computer Vision syllabus.*
