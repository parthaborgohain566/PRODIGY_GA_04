# Image-to-Image Translation with cGAN (Pix2Pix)

This notebook implements a Conditional Generative Adversarial Network (cGAN) for image-to-image translation tasks.

---
## Output
<img width="950" height="315" alt="image" src="https://github.com/user-attachments/assets/5728f656-b1da-405a-afe2-9a76b12ea023" />
---

## 📌 Project Overview

Unlike traditional unstructured GANs that generate images from random noise vectors, Pix2Pix uses a conditional GAN architecture:

* **Input Condition ($x$):** Architectural segmentation masks.
* **Target Image ($y$):** Real building facade photo.
* **Generated Image ($\hat{y}$):** Predicted building facade generated from the mask.

---

## 🏗️ Model Architecture
---
<img width="522" height="1430" alt="image" src="https://github.com/user-attachments/assets/b260c4a9-cf54-4cf8-809b-f54433a2bd06" /> <br>
<img width="376" height="814" alt="image" src="https://github.com/user-attachments/assets/a8b03b7c-cb76-4631-918c-35efb708d1a2" />


---
### 1. Generator (U-Net)

The Generator translates input masks to target images using an encoder-decoder structure with skip connections between mirrored layers to pass high-resolution spatial information across the network.

* **Encoder (Downsampling):** 8 `Conv2D` blocks with `LeakyReLU` and `Batch Normalization`.
* **Decoder (Upsampling):** 7 `Conv2DTranspose` blocks with `ReLU`, `Dropout`, `Batch Normalization`, and skip connections.
* **Output Layer:** `Conv2DTranspose` with `tanh` activation producing pixels in $[-1, 1]$.

### 2. Discriminator (70x70 PatchGAN)

The Discriminator evaluates local image structure using a $70 \times 70$ PatchGAN classifier, evaluating $70 \times 70$ patches individually across a $30 \times 30$ output feature grid.

* **Inputs:** Concatenated pair of `[Input Mask, Target Image]` or `[Input Mask, Generated Image]`.
* **Architecture:** Strided `Conv2D` layers with `LeakyReLU` and `Batch Normalization`.

---

## 📐 Loss Functions and Optimizers

* **Total Generator Loss:** $\text{GAN Loss} + (\lambda \times \text{L1 Loss})$, where $\lambda = 100$.
* **Optimizers:** Adam ($\text{learning rate} = 0.0002$, $\beta_1 = 0.5$) for both Generator and Discriminator models.

---

## 📁 Dataset and Preprocessing

* **Source:** Hugging Face Hub (`huggan/facades`).
* Images resized to $256 \times 256$ using Nearest Neighbor interpolation.
* Pixels normalized from $[0, 255]$ to $[-1, 1]$.
* Dataset shuffled with a buffer size of 400 and batched with `batch_size = 1`.

---

## 📊 Training Results Summary

* Trained for **20 epochs** (~600 steps per epoch).
* **Average epoch time:** ~75–88 seconds.
* **Saved model artifacts upon completion:**
* `generator_facades.weights.h5`
* `discriminator_facades.weights.h5`
* `pix2pix_generator.keras`



---

## 🚀 Prerequisites

Required Python libraries:

```bash
pip install tensorflow matplotlib numpy pillow datasets pydot graphviz

```
