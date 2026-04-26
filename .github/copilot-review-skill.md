# Code Review Standards

> Base review standards: [CameronImmesoete/.github/.github/copilot-review-skill.md@1f79bfb](https://github.com/CameronImmesoete/.github/blob/1f79bfb3e9eee277d05ecdd3332220204cb0f38b/.github/copilot-review-skill.md)

## Repository-Specific Review Criteria

### GAN Architecture
- Generator output shape must match discriminator input shape
- Discriminator should not overpower generator (balanced capacity)
- Activation functions appropriate per layer (ReLU/LeakyReLU in hidden, tanh/sigmoid in output)
- Batch normalization applied correctly (not on discriminator input layer)

### Training Stability
- Loss functions correct for GAN training (binary cross-entropy or Wasserstein)
- Learning rates appropriate (typically 1e-4 to 2e-4 for Adam)
- Gradient clipping if using Wasserstein loss
- No mode collapse indicators (discriminator loss near zero)

### Resource Management
- Batch size reasonable for target GPU memory
- Training GIF generation doesn't leak memory over epochs
- Dataset loading uses tf.data pipeline with prefetch
