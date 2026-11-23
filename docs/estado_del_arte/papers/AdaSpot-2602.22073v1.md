# AdaSpot: Spend Resolution Where It Matters for Precise Event Spotting

> Convertido a Markdown a partir de la versión HTML de arXiv ([arXiv:2602.22073v1](https://arxiv.org/abs/2602.22073v1)) y cotejado con el PDF [AdaSpot-2602.22073v1.pdf](AdaSpot-2602.22073v1.pdf).

- **Artur Xarles** — Universitat de Barcelona, Barcelona, Spain; Computer Vision Center, Cerdanyola del Vallès, Spain; arturxe@gmail.com
- **Sergio Escalera** — Universitat de Barcelona, Barcelona, Spain; Computer Vision Center, Cerdanyola del Vallès, Spain; Aalborg University, Aalborg, Denmark; sescalera@ub.edu
- **Thomas B. Moeslund** — Aalborg University, Aalborg, Denmark; tbm@create.aau.dk
- **Albert Clapés** — Universitat de Barcelona, Barcelona, Spain; Computer Vision Center, Cerdanyola del Vallès, Spain; aclapes@ub.edu

## Abstract

Precise Event Spotting aims to localize fast-paced actions or events in videos with high temporal precision, a key task for applications in sports analytics, robotics, and autonomous systems. Existing methods typically process all frames uniformly, overlooking the inherent spatio-temporal redundancy in video data. This leads to redundant computation on non-informative regions while limiting overall efficiency. To remain tractable, they often spatially downsample inputs, losing fine-grained details crucial for precise localization. To address these limitations, we propose **AdaSpot**, a simple yet effective framework that processes low-resolution videos to extract global task-relevant features while adaptively selecting the most informative region-of-interest in each frame for high-resolution processing. The selection is performed via an unsupervised, task-aware strategy that maintains spatio-temporal consistency across frames and avoids the training instability of learnable alternatives. This design preserves essential fine-grained visual cues with a marginal computational overhead compared to low-resolution-only baselines, while remaining far more efficient than uniform high-resolution processing. Experiments on standard PES benchmarks demonstrate that **AdaSpot** achieves state-of-the-art performance under strict evaluation metrics (*e.g*., +3.96 and +2.26 mAP$@0$ frames on Tennis and FineDiving), while also maintaining strong results under looser metrics. Code is available at: <https://github.com/arturxe2/AdaSpot>.

## 1 Introduction

![Figure 1](img/AdaSpot-2602.22073v1/top_right_v6.png)

**Figure 1:** Illustration of standard PES approaches: (a) high-resolution videos incur high computational cost, whereas (b) low-resolution videos reduce cost but lose fine-grained details crucial for precise temporal localization. In contrast, (c) **AdaSpot** captures global context from low-resolution videos and adaptively applies high-resolution processing to task-relevant regions, preserving fine-grained details efficiently.

Recent progress in action recognition [19] has enabled reliable classification of *what* happens in a video. However, many applications also require awareness of *when* it happens. Precise temporal detection –*i.e*., determining *exactly when* an action or event occurs– is crucial for tasks such as identifying decisive sports moments [29, 33], anticipating pedestrian behavior [37], and facilitating responsive human–robot interaction [18, 61]. Within this context, two established formulations are Temporal Action Localization (TAL) [54] and Event Spotting (ES) [56]. TAL models actions as temporal segments, whereas ES represents an action or event using a single keyframe. This representation makes ES well-suited for fast scenarios where events are brief. Precise Event Spotting (PES) [14, 27] further refines ES by enforcing near frame-level accuracy, therefore increasing the task’s difficulty, as even minor temporal errors can result in missed events. Although sports datasets currently dominate PES benchmarks due to their fast-paced nature and the need for high temporal precision, PES itself is domain-agnostic and broadly applicable to any setting where accurate temporal detection is critical.

Existing PES methods primarily focus on temporal modeling, exploring multi-scale representations and long-range dependencies [34, 51, 52]. However, they typically process all frames uniformly, disregarding the substantial spatio-temporal redundancy inherent in videos. This uniform processing leads to high computational costs on high-resolution inputs (Fig. 1(a)), as much computation is spent on regions with limited task relevance. To remain tractable, models are often trained on spatially downsampled videos (Fig. 1(b)), while keeping high temporal resolution to meet the task’s precision requirements. Yet, spatial downsampling can cause the loss of fine details observable only at high resolutions –details that are crucial for precise temporal detection (*e.g*., in tennis, the subtle cue of the ball contacting the ground can vanish, hindering exact frame identification). This issue is further amplified in far-view scenes, where action cues occupy only a small portion of the frame.

Prior work in video action recognition has addressed similar challenges through dynamic computation strategies [13] that adaptively allocate computational resources to task-relevant regions. A prominent line of work [23, 44–47, 62] focuses on reducing spatial redundancy at the input level by first employing lightweight modules to identify informative regions, which are then processed with higher-capacity computation. This approach effectively reduces unnecessary computation on uninformative areas while maintaining strong performance by concentrating resources where they matter most. However, most existing methods rely on learnable cropping mechanisms [44–46, 62] for region selection, which can be unstable to train even in standard action recognition settings [47]. In the case of PES, where supervision signals are weaker due to the highly localized spatio-temporal nature of events, directly applying such cropping-based approaches amplifies these instabilities, often leading to inconsistent or unreliable crops across frames (see Sec. 4.3).

To address these limitations while effectively mitigating spatial redundancy in PES, we propose **AdaSpot** (Fig. 1(c)), a simple yet effective framework that adaptively focuses computation on task-relevant regions. AdaSpot operates at multiple resolutions: it first processes full frames at low resolution to extract global task-relevant features and guide the selection of a single region of interest (RoI) for each frame. These RoIs are then processed at high resolution –*i.e*., with increased computational capacity– to capture fine-grained details, which are fused with the global features to preserve both local and global information. RoI selection is performed using saliency maps derived from the low-resolution features in a training-free manner, avoiding the instability of alternative learnable cropping mechanisms. Our RoI selector further addresses three key challenges when extracting RoIs from saliency maps: (1) it replaces zero-padding with replicate padding to mitigate *center bias*, (2) applies spatio-temporal smoothing to reduce *noisy activations* and ensure consistent RoI selection, and (3) adapts RoI size according to the saliency spread to handle *varying required RoI sizes* across datasets, action types, or camera views. This design enables AdaSpot to capture fine-grained, task-relevant details at low computational cost, since only a small portion of each frame is processed at high resolution. Our main contributions can be summarized as follows:

- To the best of our knowledge, we introduce the first PES framework that explicitly addresses spatial redundancy at the input level by adaptively allocating high-resolution processing only to the most task-relevant region of each frame. This design preserves fine-grained visual cues essential for frame-level precision, introducing only marginal overhead compared to a low-resolution-only baseline, while still incurring far less computational cost than uniform high-resolution processing.
- We propose an unsupervised, task-aware RoI selection strategy based on saliency maps, avoiding the training instability of learnable cropping alternatives. Our RoI selector mitigates activation bias and noise to ensure robust and consistent localization across frames, while dynamically adapting region sizes to varying scene and action characteristics.
- **AdaSpot** achieves state-of-the-art results across multiple PES benchmarks under tight temporal error tolerances, while improving –or, at least– maintaining competitive computational efficiency relative to prior work.

## 2 Related Work

ES addresses the problem of identifying when actions or events occur within a video. In current ES literature [12, 14], both extended actions and brief events are represented using a single keyframe. Following this convention, we use the terms *action* and *event* interchangeably, as the distinction does not affect our work.

**Event spotting.** Given their conceptual similarity, ES and TAL methods often share architectural components and can be broadly categorized into *end-to-end* and *two-stage* formulations. *End-to-end* models [10, 14, 25, 34, 42, 43, 51, 52] jointly learn visual feature extraction and temporal modeling within a unified framework, typically employing lightweight 2D backbones with local temporal modules to maintain training efficiency. In contrast, *two-stage* approaches [36, 38, 50, 58, 64] decouple these processes, first extracting video features using larger 2D or 3D encoders, followed by a separate temporal modeling stage. Temporal modeling is commonly realized through sequential architectures such as RNNs [14, 34, 42], Transformers [10, 38, 50, 58, 64], or Temporal Convolutions [36, 38, 51, 52]. To capture short- and long-range dependencies, several works adopt multi-scale processing, employing either pyramid networks [36, 58] or U-Net-like architectures [38, 51, 52].

Recent ES research increasingly favors end-to-end pipelines following E2E-Spot [14], which demonstrates that simple joint modeling can outperform multi-stage alternatives while enabling low-latency inference. Subsequent work further advances temporal modeling within this framework: T-DEED [51, 52] replaces Gate Shift Modules (GSM) [40] with Gate Shift Fuse (GSF) [41] and introduces multi-scale temporal processing; Santra et al. [34] incorporates long-range refinement modules within the backbone; and UGLF [42] adds a vision-language branch to highlight semantically salient content. Despite these advances, most methods still ignore spatio-temporal redundancy and operate on downsampled inputs, losing fine-grained detail. Instead, explicitly addressing redundancy can enable adaptive computation, focusing high-resolution processing and spending additional compute only where it matters.

**Reducing spatio-temporal redundancy.** Prior work in video action classification attempts to mitigate the spatio-temporal redundancy inherent in videos by concentrating computation on task-relevant regions. These approaches can be broadly categorized as *architecture-based* or *input-based*. *Architecture-based* methods retain full-frame processing but allocate computation unevenly across feature map locations. Examples include deformable and sparse networks [4, 7, 22, 55], which dynamically attend to salient spatial or temporal positions. However, such approaches typically still require dense early-stage processing to establish global context before selectively attending to informative regions, which constrains their efficiency gains.

*Input-based* methods reduce redundancy at the input level by first identifying relevant regions using a lightweight mechanism, which are then processed at higher resolution and/or with larger networks. Early work focused on **temporal redundancy** [11, 20, 21, 28, 48, 49, 53], either by adjusting frame resolution based on frame importance [28], selecting the most informative frames [21], or stopping inference once sufficient evidence is obtained [11]. In PES, however, temporal redundancy is harder to deal with –skipping frames risks missing entire events. In contrast, **spatial redundancy** [15–17, 23, 44–47, 62] is more suitable for the nature of the task. Early strategies used naïve center cropping [17], motion-based region selection [15], or object-based region selection [16]. More recent approaches perform task-aware region selection, as exemplified by the AdaFocus family [44–47] and CoViFocus [62], which learn per-frame crop regions through reinforcement learning or differentiable cropping. Yet, learning crop locations directly in pixel space [45, 62] remains challenging due to limited supervision, low input diversity, and training instability [47]. Learning crop positions in feature space [46, 47] alleviates some of these issues; however, applying such methods to PES remains difficult (see Sec. 4.3) because of the weaker supervision signals associated with short and highly localized PES events. Alternatively, softer approaches such as saliency-guided warping [23] expand discriminative regions while retaining global frame structure, though they can introduce geometric distortions that hinder spatio-temporal modeling.

In the context of PES, redundancy-aware methods remain underexplored. Concretely, only UGLF [42] attempts to mitigate spatial redundancy architecturally by using a vision-language model to focus on features corresponding to task-relevant concepts (*e.g*., player or ball in football). However, it requires hand-crafted dataset-specific vocabularies, does not exploit higher-resolution for relevant regions, and offers limited efficiency gains due to the overhead of the vision-language model.

We address spatial redundancy in PES with an input-based approach. To overcome the challenges of learning reliable RoIs in prior input-based methods, we introduce an unsupervised, task-aware strategy based on saliency, which mitigates training instabilities while generating semantically meaningful and temporally consistent RoIs across frames. Within PES, our method differs from UGLF in that it operates directly on the input space for greater computational efficiency, selects task-aware regions without requiring any dataset-specific vocabularies, and avoids dependence on large vision-language models.

## 3 Method

![Figure 2](img/AdaSpot-2602.22073v1/mainModel.png)

**Figure 2:** Overview of our proposed method, **AdaSpot**. (a) The framework uses a *low-resolution extractor* to process low-resolution clips and generate global features $F_{l}$ and spatial maps $F_{s}$. A *RoI selector* leverages $F_{s}$ to identify the most relevant region in each frame. The resulting RoI sequence is then processed by a *high-resolution extractor* to capture fine-grained features, $F_{h}$. $F_{l}$ and $F_{h}$ are linearly projected, aggregated, and passed through a *temporal modeler*, before a *prediction head* produces per-frame classifications. (b) Details of the RoI selector: channel averaging generates saliency maps from $F_{s}$, spatio-temporal smoothing reduces noise, and adaptive-scale RoI selection adjusts the RoI size to the saliency spread.

**Problem definition.** PES [14] aims to localize discrete actions or events within an untrimmed video $X$. The goal is to identify all event instances $E=\{e_{1},\dots,e_{N}\}$, where $N$ is the number of events and may vary across videos. Each event $e_{i}$ is defined by an event class $c_{i}\in\{1,\dots,C\}$, with $C$ the total number of event categories, and a temporal position $t_{i}$ indicating the frame corresponding to the exact or most representative moment of that action or event. Thus, each event can be represented as a pair $e_{i}=(c_{i},t_{i})$.

### 3.1 Methodology

Our proposed method, **AdaSpot**, is designed for PES, where the model adaptively identifies the most task-relevant region in each frame and processes it at high resolution, enabling the capture of fine-grained visual cues crucial for precise temporal localization. As shown in Fig. 2(a), **AdaSpot** consists of a low-resolution feature extractor, a RoI selector, a high-resolution feature extractor, a temporal modeler, and a prediction head. Videos are divided into fixed-length clips of $L$ densely sampled frames. Each clip is provided in high resolution, from which a low-resolution version is first derived and passed through the low-resolution feature extractor to: (1) capture per-frame global context features relevant to the task, and (2) produce spatially structured feature maps that guide the identification of the most informative region within each frame. The RoI selector then uses these feature maps to generate saliency maps and identify one RoI per frame, while enforcing spatio-temporal consistency across frames. The selected regions are aggregated on the fly to form a high-resolution clip of RoIs, which is processed by the high-resolution feature extractor to obtain fine-grained per-frame representations. The temporal modeler fuses features from both the low- and high-resolution branches, combining global context and local details, and refines them through a long-term temporal module to capture temporal dependencies. Finally, the prediction head classifies each frame as either an event or background.

#### 3.1.1 Low-resolution feature extractor

The low-resolution feature extractor $\phi_{l}$ operates on the full-view input sequence $X_{l}\in\mathbb{R}^{L\times W_{l}\times H_{l}\times 3}$, obtained by resizing each frame to $W_{l}\times H_{l}$. Following Hong et al. [14], we adopt RegNetY [32], a highly efficient 2D ConvNet, as our feature extractor. To capture local temporal information, GSF [41] modules are embedded into each of its bottleneck blocks. The extractor outputs global context features $F_{l}=\phi_{l}(X_{l})\in\mathbb{R}^{L\times d}$, where $d$ is the channel dimension. We also retain the feature maps from the final layer before spatial aggregation, $F_{s}\in\mathbb{R}^{L\times W_{s}\times H_{s}\times d}$, which preserve spatial structure, and use them to generate the saliency maps that guide the RoI selector.

#### 3.1.2 RoI selector

We propose a training-free RoI selection mechanism (Fig. 2(b)) that bypasses the instability of learning-based methods while producing stable, semantically meaningful regions across frames. Specifically, it leverages the intrinsic activation patterns in the low-resolution feature maps $F_{s}$ to guide the selection of the most informative regions.

**Saliency map generation.** As noted by Zhou et al. [63], activation maps from deeper convolutional layers tend to exhibit stronger responses over task-relevant regions. We exploit this by averaging $F_{s}$ along the channel dimension to obtain saliency maps $S\in\mathbb{R}^{L\times W_{s}\times H_{s}}$. Each frame map $S_{l}$, $l\in\{0,\dots,L-1\}$, is then min-max normalized for consistent scaling across frames. Since $F_{s}$ comes from deep backbone layers, its spatial resolution is heavily downsampled. Selecting RoIs directly on this coarse grid limits RoIs to a few discrete spatial positions –so even a one-cell shift can correspond to a large displacement in the original frame, causing abrupt and unstable RoI changes. To address this, we upsample $S_{l}$ by a factor $k$ along the spatial dimension. While this does not add information, it provides a denser sampling grid for RoI selection, resulting in more precise localization and smoother temporal trajectories.

**Stabilizing the saliency maps.** Extracting RoIs from $S$ involves three main challenges: (1) *Center bias:* zero-padding in convolutional layers can reduce activation strength near image borders [1], biasing RoIs toward the center; (2) *Noisy activations:* fluctuations in saliency maps may lead to spatially and temporally inconsistent RoIs across frames; and (3) *Variable RoI scale:* appropriate RoI size varies across datasets, action types, and camera views, so a fixed scale may fail to capture all relevant regions.

We address these challenges as follows. (1) *Center bias removal:* zero-padding in the low-resolution backbone $\phi_{l}$ is replaced with replicate padding, mitigating such artificial emphasis toward the center of the frames. (2) *Spatio-temporal consistency:* a spatio-temporal Gaussian smoothing filter is applied to $S$, yielding $\tilde{S}$. This reduces noise in the saliency maps and ensures they are spatio-temporally consistent, providing a stable basis for subsequent RoI selection. (3) *Scale adaptivity:* to derive RoIs that flexibly adjust to the saliency spread, we first normalize each frame map $\tilde{S}_{l}$ such that its values sum up to 1, interpreting $\tilde{S}_{l}(x,y)$ as a spatial importance probability at each location. For each frame, we select a single RoI $\mathcal{R}_{l}$, as, for the studied task and datasets, the relevant regions are typically concentrated in a single location, making multiple regions unnecessary (see Supp. C). We define $\mathcal{R}_{l}$ as the smallest rectangular region with a fixed aspect ratio that captures a cumulative importance above a threshold $\tau$, *i.e*., $\sum_{(x,y)\in\mathcal{R}_{l}}\tilde{S}_{t}(x,y)\geq\tau$, while satisfying a minimum region size $(W_{r},H_{r})$. The threshold $\tau$ controls the tightness of the resulting region and is set empirically.

The resulting set of RoIs, $\mathcal{R}=\{\mathcal{R}_{l}\}_{l=0}^{L-1}$, are cropped from the high-resolution clips $X_{h}\in\mathbb{R}^{L\times W_{h}\times H_{h}\times 3}$ and resized to a fixed size $(W_{r},H_{r})$ for efficient batched processing, yielding high-resolution RoI clips $X_{r}\in\mathbb{R}^{L\times W_{r}\times H_{r}\times 3}$. This enables **AdaSpot** to select RoIs that are task-aware, spatially unbiased, consistent across frames, and scale-adaptive, spending high-resolution computation on the most informative regions for precise event localization.

#### 3.1.3 High-resolution feature extractor

The high-resolution feature extractor $\phi_{h}$ operates on the high-resolution RoI clips $X_{r}$, producing fine-grained representations of the selected regions, *i.e*., $F_{h}=\phi_{h}(X_{r})\in\mathbb{R}^{L\times d}$. Its architecture mirrors that of $\phi_{l}$, but with independent parameters to ensure specialized low- and high-resolution feature representations.

#### 3.1.4 Temporal modeler

The temporal modeler integrates complementary information from both the low- and high-resolution branches (*i.e*., global context from $F_{l}$ and fine-grained details from $F_{h}$) while modeling longer-term temporal dependencies.

**Feature Alignment and Fusion.** Before fusion, each branch’s features are projected using lightweight two-layer MLPs with an intermediate ReLU to facilitate distributional alignment, *i.e*., $F_{l}^{\prime}=\phi_{l}^{\text{proj}}(F_{l})$ and $F_{h}^{\prime}=\phi_{h}^{\text{proj}}(F_{h})$. The aligned features are then fused via a max-pooling operation $F_{f}=max(F_{l}^{\prime},F_{h}^{\prime})\in\mathbb{R}^{L\times d}$, which is both effective and computationally efficient. As detailed in Supp. C, more complex fusion mechanisms do not yield significant improvements while increasing computational costs.

**Temporal modeling.** The fused representations $F_{f}$ are subsequently processed by a bidirectional GRU layer to capture longer-range temporal dependencies, $F_{t}=GRU(F_{f})\in\mathbb{R}^{L\times 2d}$. We adopt the GRU as our temporal model, as it has demonstrated strong performance in PES [14, 34].

#### 3.1.5 Prediction head

The prediction head produces per-frame class probabilities including a background class. A single linear layer $\phi_{\text{pred}}$ maps $F_{t}$ to logits $\hat{y}=\phi_{\text{pred}}(F_{t})\in\mathbb{R}^{L\times(C+1)}$, which are converted to probabilities via softmax.

### 3.2 Training details

Following Hong et al. [14], we formulate PES as frame-level classification. At each frame $l$, the model predicts class probabilities $\hat{y}_{l}$, which are compared against the one-hot ground-truth labels $y_{l}$ using a weighted cross-entropy loss: $\mathcal{L}_{f}=\frac{1}{L}\sum_{l=0}^{L-1}CE_{w}(y_{l},\hat{y}_{l})$, where $w$ is a scalar weight to balance foreground and background classes.

**Auxiliar supervision.** Training with only $\mathcal{L}_{f}$ is, however, unstable (see Sec. 4.3). To stabilize optimization and encourage both low- and high-resolution branches to learn discriminative and complementary features, we introduce auxiliary supervision at each branch. Specifically, we attach identical temporal modeling and prediction heads –a GRU layer followed by a linear classifier– to the low- and high-resolution feature streams, $F_{l}$ and $F_{h}$, and compute auxiliary weighted cross-entropy losses $\mathcal{L}_{l}$ and $\mathcal{L}_{h}$, respectively.

The overall loss is a weighted combination, $\mathcal{L}=\lambda_{f}\mathcal{L}_{f}+\lambda_{l}\mathcal{L}_{l}+\lambda_{h}\mathcal{L}_{h}$, with $\lambda_{f},\lambda_{l},\lambda_{h}$ controlling the contribution of each term. This formulation enforces that (i) the low-resolution branch learns stable, task-relevant features for reliable RoI selection, and (ii) the high-resolution branch captures fine-grained details that naturally complement those from the low-resolution branch. In practice, this one-stage scheme provides robust and stable end-to-end training.

### 3.3 Inference

At inference time, we use clips with 50% overlapping. Moreover, to reduce the number of candidate events, Soft Non-Maximum Suppression [3] is applied. Additionally, the auxiliary supervision modules used in the low- and high-resolution branches are discarded.

**Table 1:** Comparison with state-of-the-art methods for PES on Tennis, FineDiving, FineGym, and F3Set datasets. Bold and underlined values indicate the best and second-best results, respectively. The number of parameters (in millions) and GFLOPs for each method are also reported. For AdaSpot, we show results for two variants AdaSpot<sup>*s*</sup> and AdaSpot<sup>*b*</sup> (using RegNetY-200MF and RegNetY-400MF as feature extractors, respectively), reporting the mean over three random seeds along with the standard deviation (± std). <sup>†</sup> denotes results obtained by re-running inference with the provided checkpoints using SNMS with a window of two and 50% overlapping for a fair comparison.

|  | Tennis | Tennis | Tennis | FineDiving | FineDiving | FineDiving | FineGym | FineGym | FineGym | F3Set | F3Set | F3Set | Cost | Cost |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Model | $\delta=0$f | 1f | 2f | 0f | 1f | 2f | 0f | 1f | 2f | 0f | 1f | 2f | P(M) | GFLOPs |
| E2E-Spot<sub>200MF</sub> [14] | 69.78<sup>†</sup> | 97.01<sup>†</sup> | 97.68<sup>†</sup> | 25.00<sup>†</sup> | 69.01<sup>†</sup> | 86.24<sup>†</sup> | 17.50<sup>†</sup> | 53.44<sup>†</sup> | 63.73<sup>†</sup> | – | – | – | 4.49 | 23.13 |
| E2E-Spot<sub>800MF</sub> [14] | 70.04<sup>†</sup> | 97.31<sup>†</sup> | <u>97.86</u><sup>†</sup> | 19.47<sup>†</sup> | 65.49<sup>†</sup> | 83.83<sup>†</sup> | 17.90<sup>†</sup> | **55.05<sup>†</sup>** | 66.06<sup>†</sup> | – | – | – | 12.70 | 84.93 |
| UGLF [42] | – | – | – | – | 70.00 | **87.70** | – | 50.20 | **67.80** | – | – | – | – | – |
| T-DEED<sub>200MF</sub> [52] | 56.00<sup>†</sup> | 96.91<sup>†</sup> | 97.84<sup>†</sup> | 21.33<sup>†</sup> | 71.07<sup>†</sup> | 86.87<sup>†</sup> | 17.32<sup>†</sup> | 52.99<sup>†</sup> | 63.69<sup>†</sup> | – | – | – | 16.42 | 21.97 |
| T-DEED<sub>800MF</sub> [52] | 58.43<sup>†</sup> | <u>97.34</u><sup>†</sup> | **97.97<sup>†</sup>** | 19.63<sup>†</sup> | 69.37<sup>†</sup> | 85.50<sup>†</sup> | **18.35<sup>†</sup>** | 53.97<sup>†</sup> | 64.99<sup>†</sup> | – | – | – | 64.26 | 86.34 |
| Santra et al. [34] | 61.01 | 96.21 | 97.75 | – | – | – | 15.24 | 52.31 | <u>66.57</u> | – | – | – | 6.46 | 57.84 |
| F<sup>3</sup>ED [24] | – | – | – | – | – | – | – | – | – | 24.79 | 60.71 | 64.79 | – | – |
| **AdaSpot<sup>*s*</sup>** | <u>73.49</u> ± 1.2 | 97.28 ± 0.1 | 97.76 ± 0.1 | **27.26** ± 1.9 | <u>71.78</u> ± 0.9 | <u>87.66</u> ± 0.4 | 17.52 ± 0.1 | 54.08 ± 0.4 | 64.41 ± 0.3 | <u>53.55</u> ± 1.2 | <u>67.76</u> ± 0.8 | <u>68.41</u> ± 1.0 | 7.58 | 29.78 |
| **AdaSpot<sup>*b*</sup>** | **74.02** ± 1.4 | **97.36** ± 0.1 | 97.79 ± 0.1 | <u>27.07</u> ± 1.8 | **72.00** ± 1.2 | 87.45 ± 0.9 | <u>18.21</u> ± 0.2 | <u>54.66</u> ± 0.2 | 65.24 ± 0.2 | **55.38** ± 0.3 | **69.37** ± 0.2 | **69.94** ± 0.2 | 10.63 | 56.78 |

## 4 Experiments

### 4.1 Evaluation setup

**Datasets.** We evaluate AdaSpot on four datasets. Following Hong et al. [14], we use Tennis [60], FineDiving [57], and FineGym [35] under the PES setting, along with F3Set [24], which targets more fine-grained events. We also evaluate on SoccerNet Ball Action Spotting (SN-BAS) [9, 39] under the less strict ES setting, which requires lower temporal precision, to assess AdaSpot’s effectiveness in both scenarios. Since SN-BAS has mainly been used in challenge settings [5, 6], where results often rely on dataset-specific tricks or additional data, we introduce a standardized evaluation protocol (see Supp. A) for fair and reproducible benchmarking. While these datasets focus on sports due to their suitability for PES, AdaSpot is broadly applicable to other domains requiring high temporal precision.

**Evaluation.** We follow standard practice by training on the training split, using the validation split for early stopping, and reporting results on the test split for all datasets. Performance is measured using mean Average Precision at a temporal tolerance (mAP$@$δ). For Tennis, FineDiving, FineGym, and F3Set, we adopt a strict evaluation protocol with tolerances of $\delta\in\{0,1,2\}$ frames. For SN-BAS, we report mAP at temporal tolerances of $\delta\in\{0.5,1.0\}$ seconds.

**Implementation details.** For all datasets, videos are first set to a high-resolution format of $796\times 448$. Following [14], Tennis, FineGym, and F3Set are centrally cropped to $(W_{h},H_{h})=448\times 448$. Unless stated otherwise, low-resolution inputs are set to $(W_{l},H_{l})=\tfrac{1}{2}(W_{h},H_{h})$. Also following [14], for FineDiving, the low-resolution clips are resized to a square format by setting $W_{l}=H_{l}$. RoIs are processed at $(W_{r},H_{r})=(112,112)$. We report results for two AdaSpot variants –small and big (AdaSpot<sup>*s*</sup> and AdaSpot<sup>*b*</sup>)– which differ in model size and, hence, computational cost. AdaSpot<sup>*s*</sup> uses RegNetY-200MF and AdaSpot<sup>*b*</sup> uses RegNetY-400MF (both with GSF modules) as feature extractors $\phi_{l}$ and $\phi_{h}$. During training, we apply standard data augmentation techniques, including horizontal flipping, gaussian blur, color jitter, affine transformations, and mixup [59]. All results are averaged over three runs with different random seeds for robustness. Additional implementation details are provided in Supp. B.

### 4.2 Comparison to SOTA

We compare our method, AdaSpot, with state-of-the-art spotting approaches under both PES and ES settings, focusing on the stricter metrics (mAP$@0$f for PES and mAP$@0.5$s for ES), while also reporting results on looser metrics for completeness. Results under PES are shown in Tab. 1. In **Tennis** and **FineDiving** both AdaSpot variants (AdaSpot<sup>*s*</sup> and AdaSpot<sup>*b*</sup>) achieve state-of-the-art performance. Notably, the largest gains are on the strictest metric (mAP$@0$f), improving over the best-performing competitor by +3.98 on Tennis and +2.26 on FineDiving. These results highlight AdaSpot’s ability to capture fine-grained temporal cues crucial for precise event localization. In **FineGym**, AdaSpot also delivers strong results: AdaSpot<sup>*s*</sup> matches or exceeds methods with similar computational cost (*i.e*., E2E-Spot<sub>200MF</sub>, T-DEED<sub>200MF</sub>, and Santra et al. [34]), while AdaSpot<sup>*b*</sup> outperforms most competitors and achieves performance on par with the best method, T-DEED<sub>800MF</sub>, using 6x fewer parameters and 1.5x fewer FLOPs. On **F3Set**, AdaSpot achieves SOTA results on both strict and loose metrics, surpassing F<sup>3</sup>ED with both variants, showing strong performance on more fine-grained events. In **SN-BAS under the ES setting**, similar trends are observed (see Tab. 2). AdaSpot<sup>*s*</sup> outperforms all methods with comparable computational cost, and among higher-cost approaches only E2E-Spot<sub>800MF</sub> surpasses it. In contrast, AdaSpot<sup>*b*</sup> exceeds E2E-Spot<sub>800MF</sub> by +1.75 mAP$@0.5$s with 1.66x fewer FLOPs. Overall, AdaSpot consistently achieves strong performance across datasets, offering a superior accuracy-efficiency trade-off relative to prior work. Per-class evaluations in Supp. G show these improvements are consistent across most event categories.

**Table 2:** Comparison of AdaSpot with state-of-the-art methods for the ES setting on SN-BAS. Bold and underlined values indicate the best and second-best results. The number of parameters (in millions) and GFLOPs for each method are also reported. Results for E2E-Spot and T-DEED are obtained by integrating their model into our training pipeline. For Santra et al. [34], we implement our own version of the ASTRM module following the paper specifications due to the lack of publicly available code. For AdaSpot, we show results for two variants, AdaSpot<sup>*s*</sup> and AdaSpot<sup>*b*</sup>, and report the mean over three random seeds with standard deviation (± std).

|  | SN-BAS | SN-BAS | Cost | Cost |
| --- | --- | --- | --- | --- |
| Model | δ = 0.5s | 1s | P(M) | GFLOPs |
| E2E-Spot<sub>200MF</sub> [14] | 51.46 | 55.11 | 4.49 | 40.78 |
| E2E-Spot<sub>800MF</sub> [14] | <u>54.49</u> | <u>58.65</u> | 12.70 | 150.02 |
| T-DEED<sub>200MF</sub> [52] | 45.43 | 48.41 | 12.31 | 39.58 |
| T-DEED<sub>800MF</sub> [52] | 49.39 | 53.11 | 46.22 | 151.31 |
| Santra et al. [34] | 51.07 | 55.13 | 6.84 | 82.51 |
| **AdaSpot<sup>*s*</sup>** | 53.12 ± 1.4 | 56.82 ± 1.9 | 7.58 | 46.18 |
| **AdaSpot<sup>*b*</sup>** | **56.24** ± 0.3 | **59.82** ± 0.9 | 10.63 | 90.04 |

### 4.3 Ablations

In this section, we conduct ablation studies to validate the design choices of our approach. Specifically, we first analyze the key components of our PES framework, and then compare it with alternative redundancy-aware strategies previously proposed for action recognition. Experiments are conducted on Tennis and SN-BAS. For ablations, we adopt the AdaSpot<sup>*s*</sup> configuration and disable mixup to ensure a more stable component evaluation.

**Component analysis.** Tab. 3 summarizes the contribution of each component in AdaSpot. Tab. 3(a) compares AdaSpot with its **single-branch counterparts**. Adding the high-resolution branch to the standard low-resolution pathway improves performance on the stricter tolerances (+2.12 on Tennis and +5.04 on SN-BAS), demonstrating the value of fine-grained spatial details. While using the high-resolution branch alone already surpasses the low-resolution baseline, fusing both provides the best results, proving the effectiveness of combining global context and detailed local cues. In Tab. 3(b), we analyze the effect of **padding type**. Replacing zero-padding with reflect padding improves performance. We attribute this to the center bias introduced by zero-padding [1], which can lead to biased saliency maps and suboptimal RoI selection (see Supp. C). Reflect padding alleviates this bias, producing more informative RoIs. Tab. 3(c) examines the impact of **spatio-temporal smoothing**. Using raw activations for RoI selection degrades performance (-1.63 on Tennis, -3.03 on SN-BAS), suggesting that noisy saliency produces unstable and inconsistent RoIs. Spatio-temporal smoothing alleviates this, with temporal smoothing contributing most, highlighting the importance of temporally coherent RoIs for effective high-resolution modeling. In Tab. 3(d), we analyze **adaptive RoI scale**. On Tennis, adaptive RoIs outperform a fixed crop size (+0.49 mAP$@0$f), as relevant regions can range from close-up events requiring larger RoIs to distant events where smaller RoIs suffice. In contrast, on SN-BAS, a fixed RoI size performs best, as the distant views of the dataset make the minimum region $(W_{r},H_{r})$ sufficient to capture the relevant region. Our threshold-based selector unifies both behaviors via a single hyperparameter, allowing the model to adaptively switch between fixed (*i.e*., $\tau=0$) and adaptive scales depending on dataset characteristics. Finally, Tab. 3(e) highlights the importance of **auxiliary supervision** for stable training. Low-resolution supervision ensures the low-resolution branch learns discriminative features independently, producing reliable saliency maps for RoI selection. High-resolution supervision is equally crucial: without it, early unreliable RoIs can misdirect training, causing the model to largely ignore high-resolution information and perform near the low-resolution-only baseline. Additional analysis on alternative fusion mechanisms, crop sizes, the $\tau$ parameter, multiple regions per frame, and the possibility of reusing the backbone for both branches for improved parameter efficiency are provided in Supp. C.

**Table 3:** Ablation study of AdaSpot components on Tennis and SN-BAS, evaluating the impact of single branches, padding types, smoothing methods, fixed versus adaptive RoI scales, and auxiliary supervision. We report the mean over three random seeds along with the standard deviation (± std).

|  |  | Tennis | Tennis | Tennis | SN-BAS | SN-BAS |
| --- | --- | --- | --- | --- | --- | --- |
| **Experiment** | **Experiment** | $\delta=0$f | 1f | 2f | $\delta=0.5$s | 1s |
| AdaSpot |  | 73.30 ± 0.5 | 96.90 ± 0.1 | 97.47 ± 0.1 | 53.02 ± 0.5 | 56.43 ± 0.3 |
| (a) | *Single-branch counterparts* |  |  |  |  |  |
|  | low-res branch only | 71.18 | 96.73 | 97.42 | 47.98 | 51.69 |
|  | high-res branch only | 71.91 | 96.62 | 97.26 | 52.13 | 55.95 |
| (b) | *Padding type* |  |  |  |  |  |
|  | zero-padding | 72.15 | 96.41 | 96.98 | 51.01 | 54.51 |
| (c) | *Spatio-temporal smoothing* |  |  |  |  |  |
|  | w/o spatial smoothing | 72.63 | 96.79 | 97.37 | 49.43 | 53.04 |
|  | w/o temporal smoothing | 71.86 | 96.56 | 97.17 | 49.29 | 52.67 |
|  | w/o smoothing | 71.67 | 96.87 | 97.43 | 49.99 | 53.84 |
| (d) | *RoI scale* |  |  |  |  |  |
|  | fixed ($\tau=0$) | 72.81 | 96.69 | 97.39 | 53.02 | 56.43 |
|  | adaptive | 73.30 | 96.90 | 97.47 | 51.71 | 55.88 |
| (e) | *Auxiliary supervision* |  |  |  |  |  |
|  | w/o $\mathcal{L}_{l}$ | 72.81 | 96.91 | 97.49 | 49.71 | 53.27 |
|  | w/o $\mathcal{L}_{h}$ | 71.18 | 96.40 | 97.12 | 49.48 | 53.12 |
|  | w/o $\mathcal{L}_{l}$ & $\mathcal{L}_{h}$ | 70.49 | 96.25 | 96.96 | 49.22 | 53.17 |

**Comparison with redundancy-aware methods.** In Fig. 3, we compare AdaSpot with a single low-resolution branch baseline and representative redundancy-aware approaches for action recognition, evaluated across multiple spatial resolutions. For architecture-based methods, we evaluate deformable [7] and sparse [22] convolutions, the latter implemented in two variants: *Sparse-Learned*, where sparsity locations are learned via a gating mechanism, and *Sparse-Saliency*, where they are guided by saliency maps. Input-based methods include learnable pixel-space cropping (AdaFocus-v2 [45]), learnable feature-space cropping with variable-size regions (Uni-AdaFocus [47]), and saliency-driven frame-warping [23] (implementation details in Supp. B). Among **architecture-based** approaches, efficiency gains are limited. Deformable convolutions slightly increase computational cost relative to dense convolutions and offer only moderate improvements in low-FLOP regimes. Sparse convolutions reduce FLOPs but also degrade accuracy, resulting in a trade-off comparable to the low-resolution baseline. Among **input-based** approaches, learnable cropping perform poorly when transferred to the PES setting. As analyzed in Supp. C, selected RoIs often fail to cover task-relevant regions, introducing noise during training and reducing the utility of the high-resolution branch. We attribute these limitations of learnable cropping to inherent training instabilities previously observed in action recognition [47], which are further exacerbated in the PES setting by weaker supervision signals arising from the highly localized spatio-temporal nature of PES actions. Saliency-based frame warping also does not consistently improve the accuracy-efficiency trade-off. We hypothesize geometric distortions and temporal misalignment introduced during warping hinder spatio-temporal modeling and limit performance gains. Overall, **AdaSpot** achieves the best accuracy-efficiency trade-off, surpassing the single-branch baseline and alternative redundancy-aware methods with a simple design. For instance, on Tennis, adding our high-resolution branch to the baseline improves mAP$@0$f by +3.93, +2.14, and +2.12 for base resolutions of $112\times 112$, $168\times 168$, and $224\times 224$, respectively, at only a marginal additional computational cost of approximately +6 GFLOPs. This improvement exceeds what could be obtained by uniformly increasing the baseline resolution at the same computational cost, highlighting AdaSpot’s effectiveness. Similar trends are observed on SN-BAS, with slightly more moderate gains in low-FLOP regimes.

![Figure 3](img/AdaSpot-2602.22073v1/redundancy_T.png)

![Figure 3](img/AdaSpot-2602.22073v1/redundancy_SNBAS.png)

**Figure 3:** Comparison of AdaSpot, a single-branch baseline, and redundancy-aware alternatives across multiple spatial resolutions on Tennis (left) and SN-BAS (right). Each point corresponds to a model configuration, with GFLOPs on the x-axis, mAP on the y-axis, and point size indicating the number of parameters. Models closer to the upper-left with smaller markers achieve better accuracy-efficiency trade-offs.

### 4.4 Qualitative results

Figure 4 shows qualitative examples of the generated saliency maps and the resulting RoIs across datasets. Overall, AdaSpot produces spatially coherent and semantically meaningful RoIs with high temporal consistency. In FineDiving and FineGym, where events revolve around a main athlete, saliency consistently focuses on the athlete, yielding stable RoIs over time. In Tennis and F3Set, where events center on the ball and alternate between two players, more ambiguity is introduced; nevertheless, the model attends to event-relevant regions, with highest saliency on the ball and active player. In the more crowded SN-BAS scenes, where actions also revolve around the ball, AdaSpot effectively tracks the ball, demonstrating robustness in multi-actor scenarios. Occasional uncertainty arises in frames without clear action cues (*e.g*., Fig. 4(d), frames 2–3), but the model quickly recovers once meaningful actions resume. These qualitative observations align with our quantitative results (Sec. 4.3): high-resolution processing of selected RoIs consistently improves performance, confirming that the chosen regions capture task-relevant information.

![Figure 4](img/AdaSpot-2602.22073v1/qualitative2.png)

**Figure 4:** Qualitative visualization of the saliency maps and the corresponding RoIs selected by **AdaSpot** across all evaluated datasets: Tennis, FineDiving, FineGym, F3Set, and SN-BAS. In FineDiving and FineGym, events revolve around a main athlete, whereas in Tennis, F3Set, and SN-BAS, they revolve around the ball, which is marked with a star for clarity.

## 5 Conclusion

We presented **AdaSpot**, a PES framework that explicitly addresses spatial redundancy by allocating high-resolution processing only to the most informative region in each video frame. This approach preserves the fine-grained visual cues crucial for precise localization while maintaining strong computational efficiency. Our experiments demonstrate the importance of capturing these details for accurate PES. They also show that our unsupervised RoI selector identifies semantically meaningful and temporally consistent regions while mitigating the instability issues common in existing learnable-cropping alternatives.

**Limitations and future work.** While AdaSpot shows strong performance on current PES benchmarks, its generalization beyond sports remains to be evaluated. In particular, scenarios involving simultaneous actions –where multiple regions may be relevant within a single frame– require further study to assess AdaSpot’s adaptability to multi-RoI selection. Future work could also explore more sophisticated temporal modeling or address temporal redundancy to skip non-informative frames. Although we do not identify any direct harmful applications, AdaSpot could be adapted for privacy-sensitive tasks, motivating future work to consider safeguards to prevent unintended misuse.

**Acknowledgements.** This work has been partially supported by the Spanish project PID2022-136436NB-I00, by ICREA under the ICREA Academia programme, and is part of the REPAI project, supported by the Grundfos Foundation. Part of the work was conducted during the first author’s research stay at Game On Technologies. The authors thank Game On Technologies for hosting and supporting the project.

## References

- [1] B. Alsallakh, N. Kokhlikyan, V. Miglani, J. Yuan, and O. Reblitz-Richardson (2020) Mind the pad–cnns can develop blind spots. arXiv preprint arXiv:2010.02178.
- [2] Y. Bengio, N. Léonard, and A. Courville (2013) Estimating or propagating gradients through stochastic neurons for conditional computation. arXiv preprint arXiv:1308.3432.
- [3] N. Bodla, B. Singh, R. Chellappa, and L. S. Davis (2017) Soft-nms–improving object detection with one line of code. In Proceedings of the IEEE international conference on computer vision, pp. 5561–5569.
- [4] R. Child, S. Gray, A. Radford, and I. Sutskever (2019) Generating long sequences with sparse transformers. arXiv preprint arXiv:1904.10509.
- [5] A. Cioppa, S. Giancola, V. Somers, V. Joos, F. Magera, J. Held, S. A. Ghasemzadeh, X. Zhou, K. Seweryn, M. Kowalczyk, Z. Mróz, S. Lukasik, M. Haloń, H. Mkhallati, A. Deliège, C. Hinojosa, K. Sanchez, A. M. Mansourian, P. Miralles, O. Barnich, C. De Vleeschouwer, A. Alahi, B. Ghanem, M. Van Droogenbroeck, A. Gorski, A. Clapés, A. Boiarov, A. Afanasiev, A. Xarles, A. Scott, B. Lim, C. Yeung, C. Gonzalez, D. Rüfenacht, E. Pacilio, F. Deuser, F. S. Altawijri, F. Cachón, H. Kim, H. Wang, H. Choe, H. J. Kim, I.-M. Kim, J.-M. Kang, J. Tursunboev, J. Yang, J. Hong, J. Lee, J. Zhang, J. Lee, K. Zhang, K. Habel, L. Jiao, L. Li, M. Gutiérrez-Pérez, M. Ortega, M. Li, M. Lopatto, N. Kasatkin, N. Nemtsev, N. Oswald, O. Udin, P. Kononov, P. Geng, S. G. Alotaibi, S. Kim, S. Ulasen, S. Escalera, S. Zhang, S. Yang, S. Moon, T. B. Moeslund, V. Shandyba, V. Golovkin, W. Dai, W. Chung, X. Liu, Y. Zhu, Y. Kim, Y. Li, Y. Yang, Y. Xiao, Z. Cheng, and Z. Li (2024) SoccerNet 2024 challenges results. arXiv preprint arXiv:2409.10587. External Links: [Document](https://dx.doi.org/10.48550/arXiv.2409.10587), [Link](https://doi.org/10.48550/arXiv.2409.10587)
- [6] A. Cioppa, S. Giancola, V. Somers, F. Magera, X. Zhou, H. Mkhallati, A. Deliège, J. Held, C. Hinojosa, A. M. Mansourian, et al. (2024) SoccerNet 2023 challenges results. Sports Engineering 27 (2), pp. 24.
- [7] J. Dai, H. Qi, Y. Xiong, Y. Li, G. Zhang, H. Hu, and Y. Wei (2017) Deformable convolutional networks. In Proceedings of the IEEE international conference on computer vision, pp. 764–773.
- [8] M. Dalal, A. Xarles, A. Cioppa, S. Giancola, M. Van Droogenbroeck, B. Ghanem, A. Clapés, S. Escalera, and T. B. Moeslund (2025) Action anticipation from soccernet football video broadcasts. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 6080–6091.
- [9] A. Deliege, A. Cioppa, S. Giancola, M. J. Seikavandi, J. V. Dueholm, K. Nasrollahi, B. Ghanem, T. B. Moeslund, and M. Van Droogenbroeck (2021) Soccernet-v2: a dataset and benchmarks for holistic understanding of broadcast soccer videos. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 4508–4519.
- [10] J. Denize, M. Liashuha, J. Rabarisoa, A. Orcesi, and R. Hérault (2024) COMEDIAN: self-supervised learning and knowledge distillation for action spotting using transformers. In Proceedings of the IEEE/CVF Winter Conference on applications of computer vision, pp. 530–540.
- [11] A. Ghodrati, B. E. Bejnordi, and A. Habibian (2021) Frameexit: conditional early exiting for efficient video recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15608–15618.
- [12] S. Giancola, A. Cioppa, B. Ghanem, and M. Van Droogenbroeck (2024) Deep learning for action spotting in association football videos. arXiv preprint arXiv:2410.01304.
- [13] Y. Han, G. Huang, S. Song, L. Yang, H. Wang, and Y. Wang (2021) Dynamic neural networks: a survey. IEEE transactions on pattern analysis and machine intelligence 44 (11), pp. 7436–7456.
- [14] J. Hong, H. Zhang, M. Gharbi, M. Fisher, and K. Fatahalian (2022) Spotting temporally precise, fine-grained events in video. In European Conference on Computer Vision, pp. 33–51.
- [15] P. Jana, S. Bhaumik, and P. P. Mohanta (2021) Unsupervised action localization crop in video retargeting for 3d convnets. In TENCON 2021-2021 IEEE Region 10 Conference (TENCON), pp. 670–675.
- [16] G. Kapidis, R. Poppe, E. Van Dam, L. Noldus, and R. Veltkamp (2019) Egocentric hand track and object-based human action recognition. In 2019 IEEE SmartWorld, Ubiquitous Intelligence & Computing, Advanced & Trusted Computing, Scalable Computing & Communications, Cloud & Big Data Computing, Internet of People and Smart City Innovation (SmartWorld/SCALCOM/UIC/ATC/CBDCom/IOP/SCI), pp. 922–929.
- [17] A. Karpathy, G. Toderici, S. Shetty, T. Leung, R. Sukthankar, and L. Fei-Fei (2014) Large-scale video classification with convolutional neural networks. In Proceedings of the IEEE conference on Computer Vision and Pattern Recognition, pp. 1725–1732.
- [18] B. I. Keshinro (2023) Human activity recognition using deep learning methods for human-robot interaction. Ph.D. Thesis, North Carolina Agricultural and Technical State University.
- [19] Y. Kong and Y. Fu (2022) Human action recognition and prediction: a survey. International Journal of Computer Vision 130 (5), pp. 1366–1401.
- [20] B. Korbar, D. Tran, and L. Torresani (2019) Scsampler: sampling salient clips from video for efficient action recognition. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 6232–6242.
- [21] J. Lin, H. Duan, K. Chen, D. Lin, and L. Wang (2022) Ocsampler: compressing videos to one clip with single-step sampling. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 13894–13903.
- [22] B. Liu, M. Wang, H. Foroosh, M. Tappen, and M. Pensky (2015) Sparse convolutional neural networks. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 806–814.
- [23] H. Liu, W. Lv, J. See, and W. Lin (2022) Task-adaptive spatial-temporal video sampler for few-shot action recognition. In Proceedings of the 30th ACM International Conference on Multimedia, pp. 6230–6240.
- [24] Z. Liu, K. Jiang, M. Ma, Z. Hou, Y. Lin, and J. S. Dong (2025) F<sup>3</sup> set: towards analyzing fast, frequent, and fine-grained events from videos. arXiv preprint arXiv:2504.08222.
- [25] Z. Liu, K. Jiang, M. Ma, Z. Hou, Y. Lin, and J. S. Dong (2025) Few-shot precise event spotting via unified multi-entity graph and distillation. arXiv preprint arXiv:2511.14186.
- [26] I. Loshchilov and F. Hutter (2017) Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101.
- [27] W. McNally, K. Vats, T. Pinto, C. Dulhanty, J. McPhee, and A. Wong (2019) Golfdb: a video database for golf swing sequencing. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops, pp. 0–0.
- [28] Y. Meng, C. Lin, R. Panda, P. Sattigeri, L. Karlinsky, A. Oliva, K. Saenko, and R. Feris (2020) Ar-net: adaptive frame resolution for efficient action recognition. In European conference on computer vision, pp. 86–104.
- [29] B. T. Naik, M. F. Hashmi, and N. D. Bokde (2022) A comprehensive review of computer vision in sports: open issues, future trends and research directions. Applied Sciences 12 (9), pp. 4429.
- [30] A. Neubeck and L. Van Gool (2006) Efficient non-maximum suppression. In 18th international conference on pattern recognition (ICPR’06), Vol. 3, pp. 850–855.
- [31] A. Paszke, S. Gross, F. Massa, A. Lerer, J. Bradbury, G. Chanan, T. Killeen, Z. Lin, N. Gimelshein, L. Antiga, et al. (2019) Pytorch: an imperative style, high-performance deep learning library. Advances in neural information processing systems 32.
- [32] I. Radosavovic, R. P. Kosaraju, R. Girshick, K. He, and P. Dollár (2020) Designing network design spaces. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 10428–10436.
- [33] M. Rafiq, G. Rafiq, R. Agyeman, G. S. Choi, and S. Jin (2020) Scene classification for sports video summarization using transfer learning. Sensors 20 (6), pp. 1702.
- [34] S. Santra, V. Chudasama, P. Wasnik, and V. N. Balasubramanian (2025) Precise event spotting in sports videos: solving long-range dependency and class imbalance. In Proceedings of the Computer Vision and Pattern Recognition Conference, pp. 3163–3172.
- [35] D. Shao, Y. Zhao, B. Dai, and D. Lin (2020) Finegym: a hierarchical video dataset for fine-grained action understanding. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 2616–2625.
- [36] D. Shi, Y. Zhong, Q. Cao, L. Ma, J. Li, and D. Tao (2023) Tridet: temporal action detection with relative boundary modeling. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 18857–18866.
- [37] B. I. Sighencea, R. I. Stanciu, and C. D. Căleanu (2021) A review of deep learning-based methods for pedestrian trajectory prediction. Sensors 21 (22), pp. 7543.
- [38] J. V. Soares, A. Shah, and T. Biswas (2022) Temporally precise action spotting in soccer videos using dense detection anchors. In 2022 IEEE International Conference on Image Processing (ICIP), pp. 2796–2800.
- [39] SoccerNet (2023) SoccerNet ball action spotting. Note: <https://www.soccer-net.org/tasks/ball-action-spotting>Online; accessed 2025-13-10
- [40] S. Sudhakaran, S. Escalera, and O. Lanz (2020) Gate-shift networks for video action recognition. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 1102–1111.
- [41] S. Sudhakaran, S. Escalera, and O. Lanz (2023) Gate-shift-fuse for video action recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence 45 (9), pp. 10913–10928.
- [42] K. H. Tran, P. V. Do, N. Q. Ly, and N. Le (2024) Unifying global and local scene entities modelling for precise action spotting. In 2024 International Joint Conference on Neural Networks (IJCNN), pp. 1–8.
- [43] R. Voeikov, N. Falaleev, and R. Baikulov (2020) TTNet: real-time temporal and spatial video analysis of table tennis. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition workshops, pp. 884–885.
- [44] Y. Wang, Z. Chen, H. Jiang, S. Song, Y. Han, and G. Huang (2021) Adaptive focus for efficient video recognition. In proceedings of the IEEE/CVF international conference on computer vision, pp. 16249–16258.
- [45] Y. Wang, Y. Yue, Y. Lin, H. Jiang, Z. Lai, V. Kulikov, N. Orlov, H. Shi, and G. Huang (2022) Adafocus v2: end-to-end training of spatial dynamic networks for video recognition. In 2022 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pp. 20030–20040.
- [46] Y. Wang, Y. Yue, X. Xu, A. Hassani, V. Kulikov, N. Orlov, S. Song, H. Shi, and G. Huang (2022) Adafocusv3: on unified spatial-temporal dynamic video recognition. In European Conference on Computer Vision, pp. 226–243.
- [47] Y. Wang, H. Zhang, Y. Yue, S. Song, C. Deng, J. Feng, and G. Huang (2024) Uni-adafocus: spatial-temporal dynamic computation for video recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence.
- [48] W. Wu, D. He, X. Tan, S. Chen, and S. Wen (2019) Multi-agent reinforcement learning based frame sampling for effective untrimmed video recognition. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 6222–6231.
- [49] Z. Wu, H. Li, C. Xiong, Y. Jiang, and L. S. Davis (2020) A dynamic frame selection framework for fast video recognition. IEEE Transactions on Pattern Analysis and Machine Intelligence 44 (4), pp. 1699–1711.
- [50] A. Xarles, S. Escalera, T. B. Moeslund, and A. Clapés (2023) Astra: an action spotting transformer for soccer videos. In Proceedings of the 6th International Workshop on Multimedia Content Analysis in Sports, pp. 93–102.
- [51] A. Xarles, S. Escalera, T. B. Moeslund, and A. Clapés (2024) T-deed: temporal-discriminability enhancer encoder-decoder for precise event spotting in sports videos. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 3410–3419.
- [52] A. Xarles, S. Escalera, T. B. Moeslund, and A. Clapés (2024) T-deed revisited: broader evaluations and insights in precise event spotting.
- [53] B. Xia, W. Wu, H. Wang, R. Su, D. He, H. Yang, X. Fan, and W. Ouyang (2022) Nsnet: non-saliency suppression sampler for efficient video recognition. In European Conference on Computer Vision, pp. 705–723.
- [54] H. Xia and Y. Zhan (2020) A survey on temporal action localization. IEEE Access 8, pp. 70477–70487.
- [55] Z. Xia, X. Pan, S. Song, L. E. Li, and G. Huang (2022) Vision transformer with deformable attention. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 4794–4803.
- [56] H. Xu, A. A. Baniya, S. Well, M. R. Bouadjenek, R. Dazeley, and S. Aryal (2025) Action spotting and precise event detection in sports: datasets, methods, and challenges. arXiv preprint arXiv:2505.03991.
- [57] J. Xu, Y. Rao, X. Yu, G. Chen, J. Zhou, and J. Lu (2022) Finediving: a fine-grained dataset for procedure-aware action quality assessment. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pp. 2949–2958.
- [58] C. Zhang, J. Wu, and Y. Li (2022) Actionformer: localizing moments of actions with transformers. In European Conference on Computer Vision, pp. 492–510.
- [59] H. Zhang, M. Cisse, Y. N. Dauphin, and D. Lopez-Paz (2017) Mixup: beyond empirical risk minimization. arXiv preprint arXiv:1710.09412.
- [60] H. Zhang, C. Sciutto, M. Agrawala, and K. Fatahalian (2021) Vid2player: controllable video sprites that behave and appear like professional tennis players. ACM Transactions on Graphics (TOG) 40 (3), pp. 1–16.
- [61] M. Zhdanova, V. Voronin, E. Semenishchev, Y. Ilyukhin, and A. Zelensky (2020) Human activity recognition for efficient human-robot collaboration. In Artificial Intelligence and Machine Learning in Defense Applications II, Vol. 11543, pp. 94–104.
- [62] Z. Zheng, L. Yang, Y. Wang, M. Zhang, L. He, G. Huang, and F. Li (2023) Dynamic spatial focus for efficient compressed video action recognition. IEEE Transactions on Circuits and Systems for Video Technology 34 (2), pp. 695–708.
- [63] B. Zhou, A. Khosla, A. Lapedriza, A. Oliva, and A. Torralba (2016) Learning deep features for discriminative localization. In Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 2921–2929.
- [64] X. Zhou, L. Kang, Z. Cheng, B. He, and J. Xin (2021) Feature combination meets attention: baidu soccer embeddings and transformer based temporal detection. arXiv preprint arXiv:2106.14447.

**Supplementary Material**

In this supplementary material, we provide additional details and analyses complementing the main paper. We first describe the datasets and evaluation protocols in more depth (Appendix A). Next, we expand on the implementation details of AdaSpot and the state-of-the-art methods (Appendix B). Appendix C presents extended ablation studies, while Appendix D and Appendix E provide further discussions on efficiency and randomness analyses, respectively. In Appendix F, we investigate the sensitivity of PES methods to the choice of post-processing. Finally, Appendix G reports additional per-class and qualitative results for AdaSpot.

## Appendix A Data and evaluation protocols description

In this section, we first provide additional details about the datasets, followed by further clarification of the evaluation protocols.

### A.1 Datasets description

We evaluated AdaSpot on five datasets: Tennis [60], FineDiving [57], FineGym [35], and F3Set [24] under the Precise Event Spotting (PES) setting, as well as SoccerNet Ball Action Spotting (SN-BAS) [9, 39] under the less strict Event Spotting (ES) setting, which requires lower temporal precision. In the following, we provide additional details for each dataset.

**Tennis.** The Tennis dataset, originally introduced in Zhang et al. [60] and later extended by Hong et al. [14] consists of 3 345 video clips, each corresponding to a single tennis point, extracted from 28 matches. The videos have frame rates ranging from 25 to 30 frames per second. In total, the dataset contains 33 791 precisely annotated events across six classes –“serve”, “swing”, and “ball bounce”, each distinguished between near- and far-court–, with the class-wise distribution provided in Tab. 8. All annotated events are therefore ball-centric, indicating that the regions of interest in this dataset are predominantly around the ball’s spatial location.

**FineDiving.** The FineDiving dataset, introduced by Xu et al. [57], comprises 3 000 diving clips recorded at 25 frames per second. In total, it contains 7 010 events corresponding to transitions into somersaults, categorized into four classes –“pike”, “tuck”, “twist”, and “entry”–, with per-class frequencies provided in Tab. 9. All annotated events involve a single primary athlete, so the regions of interest are naturally centered on that athlete.

**FineGym.** The FineGym dataset, introduced by Shao et al. [35], comprises 5 374 untrimmed videos of gymnastics performances, originally recorded at frame rates between 25 and 60 frames per second. Following Hong et al. [14], the videos’ frame rates are standardized to 25-30 fps. In total, the dataset contains 80 166 events corresponding to the start and end of various gymnastics actions –such as “floor exercise turns”, “uneven bars dismounts”, and “balance beam turns”–, spanning 32 event classes, with per-class frequencies summarized in Tab. 10. As these events are athlete-centric, the regions of interest are predominantly focused on the main athlete.

**F3Set.** The F3Set dataset, introduced by Liu et al. [24] contains 11 584 video clips, each corresponding to a tennis point, extracted from 114 matches featuring 75 players. Videos have frame rates between 25 and 30 fps. In total, the dataset includes 42 846 precisely annotated events across 365 classes, each representing a combination of categories such as the player hitting the ball, court location, body side, shot type, shot direction, shot technique, player movement, and shot outcome. While F3Set is similar to Tennis, it features far more fine-grained events. We omit per-class statistics and analyses due to the large number of classes. As in Tennis, all events are ball-centric.

**SN-BAS.** The SN-BAS dataset [9, 39] consists of untrimmed videos from seven English Football League matches recorded at 25 fps. In total, the dataset contains 12 422 annotated ball-related events. The event classes include: “pass”, “drive”, “header”, “high pass”, “out”, “cross”, “throw-in”, “shot”, “ball-player block”, “player successful tackle”, “free-kick”, and “goal”, with per-class frequencies listed in Tab. 11. As the events are ball-centric, the relevant regions of interest are naturally centered around the ball.

### A.2 Evaluation protocols

For the Tennis, FineDiving, FineGym, and F3Set datasets under the PES setting, we follow the evaluation protocol proposed by Hong et al. [14], using the same training, validation, and test splits. The task is evaluated using mean Average Precision at a given temporal tolerance δ, denoted as $mAP@\delta$. For these datasets, we report results using temporal tolerances of $\delta\in\{0,1,2\}$ frames.

For SN-BAS, which has primarily been used for challenge purposes [5, 6], many existing methods rely on dataset-specific tricks, alternative data splits, or external data sources. To ensure fair and reproducible benchmarking, we introduce a standardized evaluation protocol. We adopt the original data splits, training on the four-game training set, using the one-game validation for early stopping, and reporting results on the two-game test set, while discarding the two-game challenge set with hidden ground-truth. Following [8], we exclude the “free-kick” and “goal” event classes due to their extremely low frequency –six and two examples, respectively, in the test split– which makes the metric highly sensitive to single correct or incorrect predictions. To maintain a more stable and meaningful evaluation, we remove these classes from our analysis. For this dataset, under the less strict ES setting, the task is evaluated using mean Average Precision with temporal tolerances of $\delta\in\{0.5,1\}$ seconds.

## Appendix B Implementation details

To ensure reproducibility, in this section we provide implementation details for AdaSpot, as well as those for the state-of-the-art models used in our comparisons. We also describe the adaptations required to apply redundancy-aware methods to the PES setting.

### B.1 AdaSpot

In addition to the implementation details provided in Sec. 4.1, we train AdaSpot on clips of $L=100$ frames with a batch size of 4. For the two model variants, AdaSpot<sup>*s*</sup>, and AdaSpot<sup>*b*</sup>, which use different feature-extractor sizes, the hidden dimensions are set to $d=368$ and $d=608$, respectively. The RoI selector uses an upsampling factor of $k=8$, and the threshold parameter $\tau$ is empirically tuned for each dataset. To mitigate class imbalance, the cross-entropy-loss assigns a weight of $w=5$ to positive classes, and the loss coefficients are set to $\lambda_{f}=\lambda_{l}=\lambda_{h}=\tfrac{1}{3}$. For F3Set, we use per-event-category prediction heads to handle multiple categories, with each event class probability computed as the product of its category probabilities. Each epoch consists of $5\,000$ randomly sampled clips. We train the models for 25 epochs on FineDiving, 50 epochs on Tennis and SN-BAS, and 100 epochs on the larger FineGym and F3Set datasets. Optimization is performed with AdamW [26], using a base learning rate of $8\mathrm{e}{-4}$, five warm-up epochs, and cosine learning-rate decay. Soft Non-Maximum Suppression uses a window size of 2 frames for PES and 12 for ES. The method is implemented in PyTorch [31], and all models are trained on a single NVIDIA RTX 6000 Ada Generation GPU.

### B.2 State-of-the-art models

**PES setting.** In the PES setting, we compare AdaSpot against several state-of-the-art methods. Specifically, we include <u>E2E-Spot [14]</u> in two variants –E2E-Spot<sub>200MF</sub> and E2E-Spot<sub>800MF</sub>– which use RegNetY-200MF and RegNetY-800MF as feature extractors, respectively. We also include <u>UGLF [42]</u> and <u>T-DEED [51]</u>, the latter likewise evaluated in two configurations, T-DEED<sub>200MF</sub> and T-DEED<sub>800MF</sub>, based on the same RegNetY backbones. In addition, we report results for <u>Santra et al. [34]</u>, and F<sup>3</sup>ED for F3Set. We exclude alternative approaches considered in Hong et al. [14], such as two-stage methods with pre-extracted features, due to their lower performance, focusing the comparison on end-to-end models that achieve high performance. As discussed in Appendix F, the choice of postprocessing technique can notably affect the evaluation metrics. To ensure a fair comparison, we report results for all methods using the same postprocessing procedure specified in Appendix C. For E2E-Spot and T-DEED, which originally report results with different postprocessing strategies, we run inference using their publicly available checkpoints and update the postprocessing accordingly. Santra et al. [34] already reports results using Soft-NMS with a window of 2. For UGLF, no public checkpoints are available; thus, we report the results as provided in their original paper, which uses a different postprocessing setup. Finally, for F<sup>3</sup>ED, we run inference using the publicly available checkpoints and modify the code to compute the mAP metrics not included in the original implementation. We additionally compare AdaSpot and F<sup>3</sup>ED under their native evaluation metrics in Appendix G.

**ES setting.** In the ES setting, we compare AdaSpot against <u>E2E-Spot</u> (in two variants: E2E-Spot<sub>200MF</sub> and E2E-Spot<sub>800MF</sub>), <u>T-DEED</u> (T-DEED<sub>200MF</sub> and T-DEED<sub>800MF</sub>), and <u>Santra et al. [34]</u>. Since none of these methods provide pre-trained models on SN-BAS under the evaluation protocol specified in Sec. A.2, we re-implemented them under our training pipeline. For E2E-Spot and T-DEED, we leverage their publicly available code. For T-DEED’s SGP-Mixer module, we adopt $B=2$ layers, a kernel size of $ks=9$, and a scalable factor of $r=4$, consistent with their SN-BAS experiments. For Santra et al. [34], no public code is available, so we implemented their proposed ASTRM module from scratch. All methods are trained on sequences of $L=100$ frames with a spatial resolution of $398\times 224$.

### B.3 Redundancy-aware methods

We provide additional implementation details for the comparison of AdaSpot with alternative redundancy-aware approaches in Sec. 4.3 of the main paper. We first report results for <u>AdaSpot</u> under three configurations with low-resolution inputs of $(W_{l},H_{l})=\tfrac{1}{4}(W_{h},H_{h})$, $(W_{l},H_{l})=\tfrac{3}{8}(W_{h},H_{h})$, and $(W_{l},H_{l})=\tfrac{1}{2}(W_{h},H_{h})$, as well as for a <u>single low-resolution baseline</u> that uses only the low-resolution branch under the same input resolutions. For the redundancy-aware methods, we adopt the taxonomy shown in Fig. 5, which distinguishes architecture-based methods –those that mitigate redundancy at the feature level– from input-based methods –those that address redundancy at the input level. Since AdaSpot targets spatial redundancy, which is more relevant for the PES task, we restrict our comparisons to methods that explicitly handle spatial redundancy. Specifically, we evaluate AdaSpot against: (i) <u>deformable convolutions [7]</u>, applied spatially; (ii) sparse convolutions [22], also applied spatially in two variants –one using saliency maps (<u>Sparse-Saliency</u>) and one using learned gating mechanisms (<u>Sparse-Learned</u>) to select sparsity locations; (iii) learnable pixel-space cropping (<u>AdaFocus-v2 [45]</u>); (iv) learnable feature-space cropping with variable size regions (<u>Uni-AdaFocus [47]</u>); and (v) <u>saliency-driven frame warping [23]</u>. Additional details for each approach are provided below.

![Figure 5](img/AdaSpot-2602.22073v1/RelatedWork.png)

**Figure 5:** Illustration of the taxonomy of methods addressing spatio-temporal redundancy. We categorize approaches into *architecture-based* and *input-based*, and indicate whether each method handles spatial redundancy, temporal redundancy, or both.

**Deformable convolutions.** For this approach, we adopt a simplified version of the AdaSpot architecture consisting of a single branch that processes frames at a fixed spatial resolution. Concretely, we retain one feature extractor, the temporal modeler, and the prediction head, while removing the RoI selector, the second feature extractor, the linear projectors, and the aggregation module. We then incorporate deformable convolutions into the remaining feature extractor. Specifically, all convolutions outside the initial “stem” block –so as to preserve standard dense early processing– with kernel size larger than $1\times 1$ are replaced by deformable convolutions with matching configuration. We report results for two variations of this approach, corresponding to input spatial resolutions of $398\times 224$ and $796\times 448$, which yield different computational costs.

**Sparse-Saliency.** This approach uses the same simplified architecture as the deformable-convolution baseline, but replaces the designated dense convolutions with sparse convolutions instead. The sparse activation pattern is determined using saliency maps: for each convolution, we compute a saliency map by channel-wise averaging the input features, following the procedure used in AdaSpot. We then retain the top 25% of positions within each frame’s feature maps with the highest activations as the active sparse locations. As in the deformable convolutions approach, we report results for two variants with input spatial resolutions of $398\times 224$ and $796\times 448$.

**Sparse-Learned.** This approach is similar to Sparse-Saliency, but the sparse activation pattern is learned end-to-end using a lightweight gating module that predicts per-position importance scores via a linear layer followed by a sigmoid. We then select the top 25% positions using a hard top-k during the forward pass, while employing a straight-through estimator (STE) [2] in backpropagation to allow gradients to flow through the soft scores. The resulting masked features are then processed by the convolutional layer. As in the other methods, we evaluate two variants with input spatial resolutions of $398\times 224$ and $796\times 448$.

**AdaFocus-v2.** For this approach, we adopt the same architecture as AdaSpot, replacing the RoI selector with the one proposed in Wang et al. [45]. Specifically, their RoI selector takes the feature maps $F_{s}$ as input and processes them through a series of spatial and temporal modules to produce per-frame predictions indicating the center of the region to crop. The approach is made differentiable via their learnable cropping mechanism, which incorporates a stop-gradient operation to improve training stability. We evaluate three variants of this method, corresponding to low-resolution inputs of $(W_{l},H_{l})=\tfrac{1}{4}(W_{h},H_{h})$, $(W_{l},H_{l})=\tfrac{3}{8}(W_{h},H_{h})$, and $(W_{l},H_{l})=\tfrac{1}{2}(W_{h},H_{h})$.

**Uni-AdaFocus.** This approach is analogous to the previous one, but replaces the RoI selector with the version proposed in Wang et al. [47]. Specifically, their method learns crop positions in the feature-space to improve training stability and is adapted to allow variable-size regions. As with the previous baseline, we evaluate three variants with low-resolution inputs of $(W_{l},H_{l})=\tfrac{1}{4}(W_{h},H_{h})$, $(W_{l},H_{l})=\tfrac{3}{8}(W_{h},H_{h})$, and $(W_{l},H_{l})=\tfrac{1}{2}(W_{h},H_{h})$.

**Saliency warping.** This approach uses the same AdaSpot architecture, but replaces the selected regions in the high-resolution branch with warped frames that emphasize the relevant regions, following Liu et al. [23]. We use the same saliency maps extracted for AdaSpot to guide the warping, as they provide reliable estimates of important regions. The warped frames are generated using the method proposed in Liu et al. [23]. As with other baselines, we evaluate three variants with low-resolution inputs of $(W_{l},H_{l})=\tfrac{1}{4}(W_{h},H_{h})$, $(W_{l},H_{l})=\tfrac{3}{8}(W_{h},H_{h})$, and $(W_{l},H_{l})=\tfrac{1}{2}(W_{h},H_{h})$.

## Appendix C Additional ablation studies

In this section, we extend the ablation analysis presented in Sec. 4.3 of the main paper. Specifically, we first provide a more detailed examination of the components and parameters of our proposed AdaSpot approach. We then analyze the instability of AdaSpot compared to learnable-based alternatives, and finally, we further examine and discuss the selected RoIs for AdaSpot in comparison with those of alternative redundancy-aware methods that operate in the input space.

### C.1 Extended component analysis

We extend the component analysis from Sec. 4.3 by first providing a visual examination of the *center bias* issue that arises when using zero-padding. We then report additional ablations on key components and parameters of AdaSpot, including alternative fusion strategies, different crop sizes, weight-sharing between the feature extractors of the low- and high-resolution branches, employing adaptive RoI aspect ratios, selecting multiple RoIs per frame, and analyzing the sensitivity of the $\tau$ parameter.

***Center bias*** **extended analysis.** In Sec. 4.3 of the main paper, we reported a performance drop when replacing replicate padding with zero-padding. We attribute this drop to a *center bias* introduced by zero-padding, which artificially reduces activation strength near the image borders [1]. Fig. 6 provides additional qualitative evidence for this effect by visualizing the resulting saliency maps and the RoIs selected when zero-padding is used. Although the saliency maps generally track the ball, activations near the borders are notably weaker than those at the center, leading the RoI selector to avoid choosing regions along the frame boundaries –even when the ball is located there. Consequently, the high-resolution branch receives less semantically meaningful crops, which negatively affects performance. This issue is most pronounced on the Tennis dataset and appears in certain runs, but when it arises, it can substantially degrade AdaSpot’s effectiveness. In contrast, we do not observe any such behavior when using replicate padding, across all experiments and random seeds.

![Figure 6](img/AdaSpot-2602.22073v1/zero_padding.png)

**Figure 6:** Qualitative visualization of saliency maps and the resulting selected RoIs when zero-padding is applied on the Tennis dataset. With zero-padding, the RoI selector ends up biased towards the central part of the frames.

**Table 4:** Extended ablation study of AdaSpot components on Tennis and SN-BAS, evaluating the impact of alternative fusion mechanisms, crop sizes, backbone reuse for both low- and high-resolution branches, adaptive RoI aspect ratios, and multiple RoIs per frame.

|  |  | Tennis | Tennis | Tennis | SN-BAS | SN-BAS |
| --- | --- | --- | --- | --- | --- | --- |
| **Experiment** | **Experiment** | $\delta=0$f | 1f | 2f | $\delta=0.5$s | 1s |
| AdaSpot |  | 73.30 | 96.90 | 97.47 | 53.02 | 56.43 |
| (a) | *Fusion mechanism* |  |  |  |  |  |
|  | mean | 71.26 | 96.48 | 97.07 | 50.55 | 54.56 |
|  | product | 71.88 | 96.75 | 97.37 | 51.24 | 54.95 |
|  | linear | 71.36 | 96.37 | 96.96 | 51.30 | 55.39 |
|  | frame-gated | 71.86 | 96.27 | 96.80 | 52.03 | 56.25 |
|  | channel-gated | 72.93 | 96.87 | 97.41 | 52.12 | 55.60 |
| (b) | *Crop size* |  |  |  |  |  |
|  | $56\times 56$ | 71.05 | 96.52 | 97.18 | 51.53 | 55.61 |
|  | $84\times 84$ | 72.19 | 96.66 | 97.28 | 51.16 | 55.26 |
|  | $168\times 168$ | 72.45 | 96.89 | 97.47 | 52.08 | 55.58 |
|  | $224\times 224$ | 73.02 | 96.77 | 97.28 | 50.60 | 54.66 |
| (c) | *Extractor reuse* |  |  |  |  |  |
|  | yes | 71.70 | 96.52 | 97.19 | 51.90 | 56.04 |
| (d) | *RoI aspect ratio* |  |  |  |  |  |
|  | adaptive | 71.66 | 96.62 | 97.24 | 51.61 | 54.97 |
| (e) | # RoIs per frame |  |  |  |  |  |
|  | 2 RoIs per frame | 72.06 | 96.83 | 97.35 | 49.19 | 52.70 |

**Additional component and parameter ablations for AdaSpot.** In Tab. 4(a) we compare AdaSpot’s max-based fusion of $F_{l}^{\prime}$ and $F_{h}^{\prime}$ with several **alternative fusion mechanisms**. Specifically, we evaluate: (i) *mean* –the per-position element-wise average; (ii) *product* –the element-wise (Hadamard) product; (iii) *linear* –concatenating the feature vectors along the channel dimension and projecting back to dimension $d$ through a linear layer; (iv) *frame-gated* –a per-frame gating mechanism that predicts a scalar $\alpha$ from the features and fuses them as $F_{f}=\alpha F_{l}^{\prime}+(1-\alpha)F_{h}^{\prime}$; and (v) *channel-gated* –the same gating approach but predicting channel-wise gates instead of a single scalar. As shown by the results, none of these alternatives surpass the simple max-based aggregation, with several of them also incurring additional computational overhead. Tab. 4(b) reports results for **varying crop sizes**. Reducing the crop below $112\times 112$ slightly decreases performance, likely because smaller regions either capture less content or are downsampled to lower resolution when resized to $(W_{r},H_{r})$. Increasing the crop size yields results closer to the baseline but does not surpass it, which we attribute to larger RoIs introducing extra context that is not task-relevant. Tab. 4(c) shows that **reusing extractor parameters** for both the low- and high-resolution branches still achieves strong performance with only minor drops. This shows that AdaSpot can be made more parameter-efficient, reducing total parameters by 37% while decreasing the strictest metrics by only -1.60 and -1.12 on Tennis and SN-BAS, respectively. Additionally, Tab. 4(d) evaluates **adaptive RoI aspect ratios**, where the RoI is no longer constrained to a fixed aspect ratio. Instead, here we take the rectangular region according to the saliency spread without enforcing this constraint. This modification results in a performance drop of -1.64 and -1.41 on Tennis and SN-BAS, respectively. We attribute this to the increased complexity of modeling RoIs with varying aspect ratios, which complicates training and reduces performance. In Tab. 4(e) we evaluate using **multiple RoIs per frame**. We extend AdaSpot to the multi-RoI setting by selecting a second region corresponding to the highest remaining saliency after excluding the first RoI for each frame. This results in two RoI clips that are processed through the shared high-resolution extractor and aggregated using element-wise maximum. For simplicity, a fixed region size is used in these experiments. The results show that incorporating more than one RoI consistently degrades performance, indicating that additional regions do not provide complementary information and instead introduce noise. This finding aligns with our qualitative analysis (Sec. 4.4), where saliency maps typically highlight a single dominant region, suggesting that a single RoI suffices for current PES benchmarks. While multi-RoI modeling could benefit scenarios with multiple simultaneous events, such dynamics are not present in the standard PES datasets. A more extensive study of multi-RoI extensions of AdaSpot, evaluated on datasets with concurrent events and multiple relevant regions, is therefore left for future work.

![Figure 7](img/AdaSpot-2602.22073v1/threshold.png)

**Figure 7:** Sensitivity of AdaSpot performance ($\text{mAP}@0\text{f}$ / $\text{mAP}@0.5\text{s}$) to variations in the threshold parameter $\tau$. The blue line denotes the Tennis dataset (left y-axis), while the green line denotes the SN-BAS dataset (right y-axis).

Finally, Fig. 7 analyzes the sensitivity to the **threshold parameter $\tau$**. Both datasets exhibit similar trends, with two main performance peaks, at $\tau=0$ and around $\tau=0.3$. The peak at $\tau=0$ arises from using fixed-size RoIs, which simplifies modeling in the high-resolution branch despite occasionally omitting context. As $\tau$ increases, performance initially decreases, indicating that the added contextual information does not compensate for the difficulty of modeling variable-size RoIs. Around $\tau=0.3$, the added context becomes beneficial enough to counteract this effect, producing the second peak. Beyond this range, performance declines once RoIs include excessive non-informative content while still varying in size. At $\tau=1$, we observe a small final peak, as the RoI becomes the full frame, again yielding fixed-size regions that simplify modeling –though at the cost of negating the purpose of the high-resolution branch, which now processes downsampled full-view clips.

### C.2 Instability analysis of learnable cropping

Tab. 5 presents a comparative instability analysis of AdaSpot against alternative learnable cropping methods: AdaFocus-v2 (AF-v2) and Uni-AdaFocus (Uni-AF). Across datasets, AdaSpot consistently achieves lower standard deviation, indicating more stable training. AF-v2 exhibits high variability, which is partially mitigated in Uni-AF. In addition, AdaSpot demonstrates more robust RoI selection, as reflected by higher performance when using high-resolution features only. In contrast, AF-v2 and Uni-AF produce more failure cases. These results highlight AdaSpot’s improved training stability and RoI robustness.

**Table 5:** Comparison of instability under the strictest metric for different learnable cropping methods with AdaSpot. Bold indicates best; (mean± std. across 3 runs).

|  | Low & high-res features | Low & high-res features | Low & high-res features | High-res features only | High-res features only | High-res features only |
| --- | --- | --- | --- | --- | --- | --- |
| **Dataset** | AF-v2 | Uni-AF | AdaSpot | AF-v2 | Uni-AF | AdaSpot |
| Tennis | 70.6 ± 1.5 | 70.2 ± 1.0 | **73.3** ± 0.5 | 68.8 ± 0.9 | 65.9 ± 0.9 | **71.9** ± 0.4 |
| SN-BAS | 49.0 ± 2.0 | 49.6 ± 1.3 | **53.0** ± 0.5 | 23.1 ± 15.1 | 39.4 ± 3.3 | **52.1** ± 0.7 |

### C.3 Qualitative RoI comparison

Fig. 8 compares the RoIs selected by our AdaSpot with those produced by input-based alternatives, specifically AdaFocus-v2 and Uni-AdaFocus. As shown, these alternative methods frequently fail to capture task-relevant regions, introducing noise during training and diminishing the effectiveness of the high-resolution branch –ultimately leading to the performance drops reported in Sec. 4.3 of the main paper. On Tennis, both AdaFocus-v2 and Uni-AdaFocus tend to converge to largely static corner crops, likely because some actions commonly occur near those areas. On SN-BAS, the crops move more dynamically, and Uni-AdaFocus localizes relevant regions more reliably (*e.g*., around the ball). However, its adaptive region size often saturates to the maximum allowed area, causing fine-grained details to be lost after resizing to $(W_{r},H_{r})$. In contrast, AdaSpot consistently selects stable, semantically meaningful RoIs. As discussed in the main paper, we attribute the limitations of such learnable-cropping approaches to the training instabilities identified in prior work [47], which our training-free RoI selector inherently avoids.

![Figure 8](img/AdaSpot-2602.22073v1/qualitativeComparison.png)

**Figure 8:** Qualitative comparison of the RoIs selected by AdaSpot, AdaFocus-v2, and Uni-AdaFocus on the Tennis (left) and SN-BAS (right) datasets. For visualization, we mark the ball position in each frame with a star, as actions in these datsets occur around the ball; thus, relevant RoIs should contain or closely surround it.

## Appendix D Efficiency analysis

In this section, we extend the efficiency analysis presented in Sec. 4.2 of the main paper. Tab. 6 reports the number of parameters and GFLOPs required to process a single clip under both the PES setting (base resolution $224{\times}224$) and the ES setting (base resolution $398{\times}224$). For Santra et al. [34], these values are derived from our re-implementation of their ASTRM module, which may therefore differ slightly from those originally reported. We exclude UGLF [42] from the comparison due to missing details regarding their vision-language module in the released code. While E2E-Spot<sub>200MF</sub>, T-DEED<sub>200MF</sub>, and Santra et al. [34] all employ RegNetY-200MF as their base extractor, E2E-Spot stands out among the most efficient in both parameter count and GFLOPs. T-DEED exhibits comparable computational cost but is substantially more parameter-intensive due to its SGP-Mixer module used for temporal modeling. In contrast, Santra et al. [34] maintains a low parameter count but incurs higher GFLOPs because the ASTRM module is inserted early in the backbone, where feature maps are still high resolution, thereby increasing the overall computational cost. AdaSpot<sup>*s*</sup>, which uses the same base extractor, introduces only a marginal increase in parameters and GFLOPs relative to E2E-Spot<sub>200MF</sub>. The additional parameters arise from duplicating the extractor for the high-resolution branch, while the extra computation stems from processing the RoI clips through this branch –adding approximately 6 GFLOPs for our standard $112\times 112$ RoI configuration. This small overhead enables AdaSpot to preserve fine-grained details and yields substantial performance improvements (see Sec. 4.2 of the main paper), resulting in a stronger efficiency-accuracy trade-off. When comparing larger extractor configurations –E2E-Spot<sub>800MF</sub>, T-DEED<sub>800MF</sub>, and AdaSpot<sup>*b*</sup>– we observe that AdaSpot<sup>*b*</sup>, despite using a smaller backbone (RegNetY-400MF) and thus being more efficient, still achieves state-of-the-art performance across both PES and ES datasets (see Sec. 4.2 of the main paper). Additionally, for AdaSpot, inference on a single clip requires only 1.97GB of GPU memory, enabling inference even on small GPUs.

**Table 6:** Efficiency comparison of AdaSpot with state-of-the-art methods in both the PES setting (typically using $224{\times}224$ inputs) and the ES setting (using $398{\times}224$ inputs). For each configuration, we report the number of parameters (in millions) and the computational cost in GFLOPs.

|  | PES | PES | ES | ES |
| --- | --- | --- | --- | --- |
| Model | P(M) | GFLOPs | P(M) | GFLOPs |
| E2E-Spot<sub>200MF</sub> [14] | 4.49 | 23.13 | 4.49 | 40.78 |
| E2E-Spot<sub>800MF</sub> [14] | 12.70 | 84.93 | 12.70 | 150.02 |
| T-DEED<sub>200MF</sub> [52] | 16.42 | 21.97 | 12.31 | 39.58 |
| T-DEED<sub>800MF</sub> [52] | 64.26 | 86.34 | 46.22 | 151.31 |
| Santra et al. [34] | 6.46 | 57.84 | 6.84 | 82.51 |
| **AdaSpot<sup>*s*</sup>** | 7.58 | 29.78 | 7.58 | 46.18 |
| **AdaSpot<sup>*b*</sup>** | 10.63 | 36.78 | 10.63 | 90.04 |

## Appendix E Randomness analysis

Training deep neural networks involves multiple sources of randomness (*e.g*., data sampling, weight initialization, and data augmentation), which can lead to noticeable performance variability across runs. Despite this, most PES methods report results from a single training run, due to the substantial computational cost of these pipelines. This practice can produce benchmarks that are sensitive to run-to-run fluctuations, making claims difficult to verify or reproduce. To provide more robust evaluations, we report results over three runs using different random seeds. For all experiments, we report the mean performance, and for the main results, we additionally provide the standard deviation to reflect variability across runs. While more runs would allow more rigorous statistical analysis –three runs are insufficient for reliable significance testing– the high computational demands of PES frameworks make extensive multi-run evaluation impractical. Nevertheless, our three-run reporting offers improved robustness over the single-run convention used in prior work.

## Appendix F Post-processing analysis

In this section, we analyze the sensitivity of PES methods to the choice of post-processing. Tab. 7 presents AdaSpot results on the Tennis dataset under different post-processing configurations. Specifically, we compare standard Non-Maximum Suppression (NMS) [30], and Soft Non-Maximum Suppression (Soft-NMS) [3], evaluating multiple window sizes $\omega\in\{1,2,3,4,5\}$. As shown, Soft-NMS generally outperforms NMS for the strictest metric (mAP$@0$f) while achieving comparable results for looser metrics. Within Soft-NMS, smaller window sizes slightly favor stricter metrics, whereas larger windows benefit more relaxed metrics. Following prior work [34, 51], we adopt a configuration that balances performance across all tolerances and use Soft-NMS with $\omega=2$. For ES experiments, where more relaxed evaluation protocols are used, larger window sizes are preferable; in this case, we find $\omega=12$ to offer the best trade-off. To ensure fair comparisons with state-of-the-art methods (Sec. 4.2 main paper), whenever possible, all results are re-extracted using the same post-processing settings.

**Table 7:** Post-processing sensitivity analysis on the Tennis dataset. We report results for standard NMS and Soft-NMS using different window sizes $\omega$. Bold and underlined values indicate the best and second-best results.

|  |  | Tennis | Tennis | Tennis |
| --- | --- | --- | --- | --- |
| **Post-processing** | **Post-processing** | $\delta=0$f | 1f | 2f |
| NMS [30] | $\omega=1$ | 62.82 | **96.93** | <u>97.61</u> |
|  | $\omega=2$ | 62.45 | 96.61 | **97.64** |
|  | $\omega=3$ | 62.35 | 96.11 | 97.58 |
|  | $\omega=4$ | 62.32 | 95.94 | 97.47 |
|  | $\omega=5$ | 62.29 | 95.84 | 97.29 |
| Soft-NMS [3] | $\omega=1$ | **75.05** | 96.02 | 96.50 |
|  | $\omega=2$ | <u>73.30</u> | 96.90 | 97.47 |
|  | $\omega=3$ | 71.53 | <u>96.92</u> | 97.56 |
|  | $\omega=4$ | 70.35 | 96.81 | 97.60 |
|  | $\omega=5$ | 69.36 | 96.70 | 97.59 |

## Appendix G Additional results and visualizations

In this section, we first analyze the per-class performance of AdaSpot in comparison with other state-of-the-art methods (E2E-Spot and T-DEED), and provide an approximate per-class evaluation of the RoI selection. We then present additional results, including F3Set evaluations under their proposed metrics, as well as visualizations of the generated saliency maps, the selected RoIs, and the corresponding model predictions for AdaSpot.

### G.1 Per-class results

Here, we report per-class results of AdaSpot compared with E2E-Spot and T-DEED, using the best-performing version of each model for each dataset. On the **Tennis** (Tab. 8) and **FineDiving** (Tab. 9) datasets, we observe the same trend as in the aggregated results from the main paper, with AdaSpot outperforming the other two methods across all event classes. In Tennis, the most notable improvements over E2E-Spot occur on “far-court swings” and “far-court serves”, highlighting that AdaSpot is particularly effective for far-view actions where uniform resolution downsampling can hinder performance. By focusing higher-resolution attention on relevant regions, AdaSpot better captures these challenging events. On **FineGym** (Tab. 10), the performance across methods is generally similar, but AdaSpot maintains competitive results across all classes, achieving strong overall performance. Finally, on **SN-BAS** (Tab. 11), AdaSpot again demonstrates superiority, achieving the best results for all but two classes. These results confirm that the improvements introduced by AdaSpot are consistent across most event categories, reinforcing its general effectiveness.

**Table 8:** Per-class analysis on the Tennis dataset. For each event class, we report the total number of observations and the AP$@0$f results for the best-performing versions of E2E-Spot, T-DEED, and AdaSpot, as well as for an AdaSpot variant using high-resolution features only (HR-only). Event classes are sorted in descending order of observations. The best result per class is highlighted in bold, and the second-best is underlined.

|  |  | AP ($\delta=0$f) | AP ($\delta=0$f) | AP ($\delta=0$f) | AP ($\delta=0$f) |
| --- | --- | --- | --- | --- | --- |
| Event | Nº observations | E2E-Spot | T-DEED | **AdaSpot** | HR-only |
| Far-court ball bounce | 8150 | <u>76.91</u> | 59.47 | **77.20** | 75.38 |
| Near-court ball bounce | 8127 | <u>76.79</u> | 68.19 | **78.98** | 78.91 |
| Far-court swing | 7123 | <u>53.24</u> | 41.52 | **64.76** | 58.77 |
| Near-court swing | 7044 | <u>56.42</u> | 48.85 | **58.83** | 58.49 |
| Near-court serve | 1690 | <u>76.79</u> | 67.91 | **79.24** | 78.67 |
| Far-court serve | 1657 | <u>80.10</u> | 64.66 | **85.09** | 81.95 |

**Table 9:** Per-class analysis on the FineDiving dataset. For each event class, we report the total number of observations and the AP$@0$f results for the best-performing versions of E2E-Spot, T-DEED, and AdaSpot. Event classes are sorted in descending order of observations. The best result per class is highlighted in bold, and the second-best is underlined.

|  |  | AP ($\delta=0$f) | AP ($\delta=0$f) | AP ($\delta=0$f) |
| --- | --- | --- | --- | --- |
| Event | Nº observations | E2E-Spot | T-DEED | **AdaSpot** |
| Entry | 2984 | 22.51 | <u>24.02</u> | **26.74** |
| Som(s).Pike | 2152 | <u>27.21</u> | 23.14 | **27.58** |
| Som(s).Tuck | 1071 | <u>31.70</u> | 21.72 | **32.23** |
| Twist(s) | 803 | <u>18.60</u> | 16.45 | **22.51** |

**Table 10:** Per-class analysis on the FineGym dataset. For each event class, we report the total number of observations and the AP$@0$f results for the best-performing versions of E2E-Spot, T-DEED, and AdaSpot. Event classes are sorted in descending order of observations. The best result per class is highlighted in bold, and the second-best is underlined.

|  |  | AP ($\delta=0$f) | AP ($\delta=0$f) | AP ($\delta=0$f) |
| --- | --- | --- | --- | --- |
| Event | Nº observations | E2E-Spot | T-DEED | **AdaSpot** |
| Uneven bars circles start | 6612 | **11.32** | 10.26 | <u>10.28</u> |
| Uneven bars circles end | 6612 | **20.19** | <u>19.89</u> | 19.63 |
| Balance beam leap_jump_hop start | 4787 | 17.52 | **19.72** | <u>18.07</u> |
| Balance beam leap_jump_hop end | 4787 | 10.31 | **12.64** | <u>10.81</u> |
| Balance beam flight_salto start | 4187 | 19.84 | <u>22.72</u> | **24.35** |
| Balance beam flight_salto end | 4187 | 6.86 | **7.76** | <u>7.48</u> |
| Uneven bars transition_flight start | 3389 | **29.86** | <u>29.60</u> | 26.65 |
| Uneven bars transition_flight end | 3389 | **30.73** | 26.99 | <u>28.09</u> |
| Floor exercise leap_jump_hop start | 3238 | **27.32** | <u>26.14</u> | 25.43 |
| Floor exercise leap_jump_hop end | 3238 | **16.41** | 14.10 | <u>14.91</u> |
| Floor exercise back_salto start | 2978 | **35.88** | 33.26 | <u>34.86</u> |
| Floor exercise back_salto end | 2978 | **13.61** | 11.95 | <u>12.83</u> |
| Balance beam flight_handspring start | 2893 | 17.64 | **19.93** | <u>19.08</u> |
| Balance beam flight_handspring end | 2893 | 23.91 | **28.80** | <u>26.50</u> |
| Vault (timestamp 0) | 2031 | **2.53** | 1.90 | <u>2.36</u> |
| Vault (timestamp 1) | 2031 | **22.54** | <u>22.53</u> | 20.15 |
| Vault (timestamp 2) | 2031 | 35.28 | <u>39.90</u> | **41.80** |
| Vault (timestamp 3) | 2031 | 5.43 | **7.07** | <u>6.29</u> |
| Uneven bars flight_same_bar start | 1624 | <u>27.30</u> | **27.85** | 25.63 |
| Uneven bars flight_same_bar end | 1624 | 26.50 | **27.58** | <u>26.82</u> |
| Balance beam turns start | 1371 | <u>12.47</u> | **13.98** | 11.56 |
| Balance beam turns end | 1371 | <u>4.67</u> | **5.33** | 4.64 |
| Floor exercise from_salto start | 1345 | 26.48 | <u>26.83</u> | **29.60** |
| Floor exercise from_salto end | 1345 | **8.97** | 8.52 | <u>8.70</u> |
| Uneven bars dismounts start | 1227 | **34.37** | <u>33.50</u> | 33.03 |
| Uneven bars dismounts end | 1227 | **10.65** | <u>8.55</u> | 7.80 |
| Balance beam dismounts start | 1218 | 21.70 | **34.94** | <u>27.86</u> |
| Balance beam dismounts end | 1218 | **7.11** | <u>6.89</u> | 4.96 |
| Floor exercise turns start | 1103 | 9.07 | **12.53** | <u>11.41</u> |
| Floor exercise turns end | 1103 | 11.07 | **15.30** | <u>13.52</u> |
| Floor exercise side_salto start | 49 | **22.35** | 8.49 | <u>19.80</u> |
| Floor exercise side_salto end | 49 | <u>2.86</u> | 1.59 | **7.70** |

**Table 11:** Per-class analysis on the SN-BAS dataset. For each event class, we report the total number of observations and the AP$@0.5$s results for the best-performing versions of E2E-Spot, T-DEED, and AdaSpot, as well as for an AdaSpot variant using high-resolution features only (HR-only). Event classes are sorted in descending order of observations. The best result per class is highlighted in bold, and the second-best is underlined.

|  |  | AP ($\delta=0.5$s) | AP ($\delta=0.5$s) | AP ($\delta=0.5$s) | AP ($\delta=0.5$s) |
| --- | --- | --- | --- | --- | --- |
| Event | Nº observations | E2E-Spot | T-DEED | **AdaSpot** | HR-only |
| Pass | 4985 | <u>85.15</u> | 83.44 | **85.94** | 85.11 |
| Drive | 4300 | <u>81.55</u> | 77.25 | **81.77** | 81.50 |
| High pass | 761 | **79.30** | 76.33 | <u>78.68</u> | 75.18 |
| Header | 713 | <u>68.14</u> | 54.27 | **68.87** | 62.62 |
| Ball out of play | 551 | 16.97 | <u>19.79</u> | **23.01** | 21.15 |
| Throw-in | 362 | <u>67.74</u> | 58.48 | **70.52** | 65.40 |
| Cross | 261 | <u>64.27</u> | 62.64 | **69.66** | 52.65 |
| Ball player block | 223 | <u>24.94</u> | 16.46 | **24.98** | 21.34 |
| Shot | 169 | <u>51.23</u> | 44.31 | **55.21** | 53.21 |
| Player successful tackle | 74 | **5.64** | 0.92 | <u>3.77</u> | 1.84 |

### G.2 Per-class RoI analysis

Per-class RoI analysis is limited by the lack of ground-truth RoIs. However, evaluating an AdaSpot variant that uses only high-resolution features provides an approximate measure of RoI precision for each event class. We report these values for Tennis and SN-BAS in Tab. 8 and Tab. 11. As shown in the tables, in Tennis, the largest performance drops compared to the full AdaSpot model occur for far-court events, indicating less precise RoI selection for distant actions. In contrast, close-court events show performance near the baseline, suggesting accurate RoI selection for nearby events. For SN-BAS, the most pronounced effect is observed for the cross event, which depends not only on the player interacting with the ball but also on the broader context of where the ball is headed, which is not covered by the RoI.

### G.3 F3Set additional evaluation

Tab. 12 presents a further evaluation on the F3Set dataset. Both AdaSpot variants outperform F<sup>3</sup>ED across all mAP metrics and the F1 score, with substantial gains. However, on the Edit score, AdaSpot performs slightly lower, highlighting the contribution of the additional context refinement module introduced in F<sup>3</sup>ED. Overall, these results demonstrate that AdaSpot achieves strong performance even on the more fine-grained event classes featured in F3Set.

**Table 12:** Comparison of AdaSpot with F<sup>3</sup>ED on the F3Set dataset using standard PES mAP metrics, as well as F1 and Edit scores. Results show the mean over three random seeds with the corresponding standard deviation (± ). Bold and underlined values indicate the best and second-best results.

|  | mAP$@0$f | mAP$@1$f | mAP$@2$f | F1<sub>*evt*</sub> | Edit |
| --- | --- | --- | --- | --- | --- |
| F<sup>3</sup>ED [24] | 24.8 | 60.7 | 64.8 | 40.3 | **74.0** |
| **AdaSpot<sup>*s*</sup>** | <u>53.55 ± 1.2</u> | <u>67.76 ± 0.8</u> | <u>68.41 ± 1.0</u> | <u>48.8 ± 1.1</u> | 72.6 ± 0.4 |
| **AdaSpot<sup>*b*</sup>** | **55.38 ± 0.3** | **69.37 ± 0.2** | **69.94 ± 0.2** | **51.66 ± 0.6** | <u>73.66</u> ± 0.4 |

### G.4 Qualitative results

**Saliency maps and selected RoIs.** To complement the visualizations in Sec. 4.4, we provide in the Supplementary Material two example clips per dataset corresponding to the best-performing AdaSpot version, showing both the saliency maps and the selected RoIs. In **Tennis** (*Video_SaliencyRoIs_Tennis_1.mp4* and *Video_SaliencyRoIs_Tennis_2.mp4*), as previously discussed, events revolve around the ball. We observe that, in most frames, the areas of highest saliency –and consequently the selected RoIs– align closely with the ball’s position. In a few cases, such as when the ball is in the air with multiple frames before or after an action, saliency occasionally shifts toward the players. However, as the clip progresses and approaches an action, the saliency consistently returns to the ball. Additionally, the generated RoIs move smoothly across frames, which facilitates effective spatio-temporal modeling within the high-resolution extractor. In **FineDiving** (*Video_SaliencyRoIs_FineDiving_1.mp4* and *Video_SaliencyRoIs_FineDiving_2.mp4*), where events center on a single athlete performing a dive, the RoIs consistently capture the athlete in nearly all frames while moving smoothly throughout the clip, demonstrating robust and reliable RoI localization. In **FineGym** (*Video_SaliencyRoIs_FineGym_1.mp4* and *Video_SaliencyRoIs_FineGym_2.mp4*), events again focus on a single athlete, and the saliency maps reliably highlight regions including the athlete. However, the camera views in this dataset are more varied, with some closer shots resulting in RoIs that cover only part of the athlete. In these cases, the selected regions tend to focus on the most relevant parts for the event (*e.g*., the hands contacting the vault during a vault, or the feet and floor when landing from a jump). We hypothesize that such closer views may explain why AdaSpot achieves slightly more modest results on this dataset, as the downsampled full-view frames already contain much of the necessary fine-grained detail. In **F3Set** (*Video_SaliencyRoIs_F3Set_1.mp4* and *Video_SaliencyRoIs_F3Set_2.mp4*), which resembles the Tennis dataset, we observe similar patterns: the highest saliency and selected RoIs closely align with the ball’s position. Finally, in **SN-BAS** (*Video_SaliencyRoIs_SNBAS_1.mp4* and *Video_SaliencyRoIs_SNBAS_2.mp4*), events are again ball-centric. In both clips, the saliency maps and selected RoIs consistently follow the ball, producing semantically meaningful regions. Only in frames without nearby events, saliency spreads more evenly across the scene, occasionally resulting in the ball falling outside the RoI.

**AdaSpot predictions.** We additionally provide, in the Supplementary Material, one example clip per dataset showing AdaSpot’s predictions, together with the temporal distance errors relative to the corresponding ground-truth annotations. In **Tennis** (*Video_Predictins_Tennis.mp4*), the strong performance reported in Tab. 1 of the main paper is clearly reflected visually: all actions are detected with high temporal precision. In **FineDiving** (*Video_Predictins_FineDiving.mp4*), all actions are still correctly identified, although some exhibit larger temporal localization errors. In **FineGym** (*Video_Predictins_FineGym.mp4*), we observe occasionally multiple predictions around a single ground-truth event, which stem from the ambiguity in localizing certain event types. Finally, in **SN-BAS** (*Video_Predictins_SNBAS.mp4*), predictions generally follow the ground truth well, achieving good precision under the more relaxed ES evaluation setting, with only one missed action near the end of the clip.
