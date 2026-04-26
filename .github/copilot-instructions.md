# Copilot Instructions

> Base instructions: [CameronImmesoete/.github/.github/copilot-instructions.md@1f79bfb](https://github.com/CameronImmesoete/.github/blob/1f79bfb3e9eee277d05ecdd3332220204cb0f38b/.github/copilot-instructions.md)

## Repository-Specific Guidelines

This is a DCGAN (Deep Convolutional Generative Adversarial Network) implementation in TensorFlow 2 with training GIF generation.

- Generator and discriminator architectures must maintain compatible tensor shapes
- Training loop stability is critical (monitor for mode collapse, vanishing gradients)
- Use TensorFlow 2 patterns (Keras API, tf.function, GradientTape)
- GPU memory management: batch size and tensor dtypes affect memory usage
- Reproducibility: random seeds should be configurable
