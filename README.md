## Image Denoiser with Supervised and Self‑Supervised Learning

This repository implements an image denoising pipeline using a two‑stage approach:

1. **Foundation Model (Self‑Supervised)**
   * Learns to restore distorted images via a self‑supervised SSLearner on top of a FoundationISP backbone.
2. **Denoiser Model (Supervised)**
   * Leverages the pretrained foundation backbone to denoise images corrupted with additive white Gaussian noise.

---

### 🔧 Requirements

* Python 3.8+
* PyTorch
* Lightning (formerly `pytorch_lightning`)
* Kornia
* Torch‑DCT
* PyTorch-Metric-Learning
* SciPy, scikit‑image, imageio

Install dependencies:

```bash
pip install torch lightning kornia torch-dct pytorch-metric-learning scipy scikit-image imageio
```

---

### 📁 Repository Structure

```plaintext
.
├── dataloader.py            # Dataset classes for distortion & noise
├── extract_foundation.py    # Trace & export foundation backbone
├── extract_denoiser.py      # Trace & export denoiser model
├── train_foundation.py      # Train self‑supervised foundation
├── train_denoiser.py        # Train supervised denoiser
├── test.py                  # Run denoising benchmarks (PSNR/SSIM)
├── visualize_featuremaps.py # Visualize learned feature channels
├── models/                  # Model definitions
│   ├── Foundation.py
│   ├── Denoiser.py
│   ├── SLearner.py
│   └── utils.py
├── trainers/                # Lightning modules
│   ├── FoundationTrainer.py
│   └── DenoiserTrainer.py
└── dataset/                 # Your train/val datasets (user-provided)
    ├── train/               # Clean images for training
    └── val/                 # Clean images for validation
```

---

### 🚀 Quickstart

Run these commands in sequence to train, export, and evaluate your denoiser:

1. **Train Foundation (Self‑Supervised)**

   ```bash
   python ./train_foundation.py
   ```

   * Outputs checkpoint under `./foundation_ckpt/lightning_logs/...`
2. **Export Foundation Backbone**

   ```bash
   python ./extract_foundation.py
   ```

   * Saves `foundation_ckpt/foundation.pth` as a TorchScript model.
3. **Train Denoiser (Supervised)**

   ```bash
   python ./train_denoiser.py
   ```

   * Uses the exported foundation model internally. Checkpoints under `./denoiser_ckpt/lightning_logs/...`
4. **Export Denoiser Model**

   ```bash
   python ./extract_denoiser.py
   ```

   * Saves `denoiser_ckpt/denoiser.pth` as a TorchScript model.
5. **Evaluate / Test Denoiser**

   ```bash
   python ./test.py
   ```

   * Runs inference on standard AWGN benchmarks (e.g., Kodak, CBSD68) at noise levels 15/25/50.
   * Reports PSNR & SSIM and saves denoised outputs under `./test/denoise/awgn`.
6. **Visualize Feature Maps (Optional)**

   ```bash
   python ./visualize_featuremaps.py
   ```

   * Displays the 32 learned channels of the foundation backbone for a sample image.

---

### ⚙️ Configuration

* Paths and hyperparameters (batch size, epochs, image shape) can be modified at the top of each `train_*.py` file.
* To resume training, set `CONTINUE_TRAINING` to a checkpoint path in either train script.

---

### 📝 License

This project is licensed under the MIT License. See the [LICENSE](https://chatgpt.com/g/g-p-6896677fb4088191972fb9a9f3e829eb-dn/c/LICENSE) file for details.

```text
MIT License

Copyright (c) 2025 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
