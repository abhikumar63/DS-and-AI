# Eigenfaces: A NumPy-Native Visual Tokenizer

Modern Multimodal LLMs use complex encoders to compress images into vector embeddings. Before reaching for PyTorch or heavy deep learning frameworks, I wanted to build the mathematical precursor to those systems: an image compressor built entirely from scratch using pure Linear Algebra (SVD/PCA) in NumPy.

## Architecture & Pipeline

This project breaks down dimensionality reduction into four executable steps:

1. **Vectorization:** Flattening 64x64 grayscale images from the Olivetti Faces dataset into 4,096-dimensional arrays.
2. **Spatial Similarity:** Calculating Cosine Similarity from scratch to mathematically prove relationships between faces in high-dimensional space.
3. **Eigenface Extraction:** Extracting the "DNA" of the dataset via Singular Value Decomposition (mean-centering the data and isolating the principal components).
4. **Latent Reconstruction:** Reconstructing identities using $k \le 50$ latent weights, achieving an ~80x compression ratio while maintaining the core visual signal.

## Visual Proof

![Reconstruction showing k=5, k=20, k=50, k=150](assets/reconstruction.png)

## Local Setup

Clone the repository and set up your local environment. 

```zsh
git clone [https://github.com/yourusername/eigenfaces-numpy.git](https://github.com/yourusername/eigenfaces-numpy.git)
cd eigenfaces-numpy

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install required dependencies
pip install numpy matplotlib scikit-learn jupyter
