# LEAR Framework on MNIST: Generating Counterfactual Explanations and validating it

This project implements the **Learn Explain Reinforce (LEAR)** framework on the MNIST dataset using PyTorch to generate counterfactual explanations. If you want to know more about counterfactual informations, please see the counterfactual-explanations repo which i have uploaded. 

Unlike typical XAI methods, LEAR modifies the **latent space** of input representations to produce realistic, minimal transformations that lead to different outcomes, enabling deep insight into model decision boundaries.

---

## Key Features

- Trains a complex convolutional classifier on MNIST digits
- Has an encoder-decoder-discriminator architecture to produce counterfactual samples
- Takes into account various loss functions like L1/Total Variation losses and custom constraints
- Saves outputs for multiple counterfactual classes (not just 1 alternate prediction)

---

## Architecture Overview

The pipeline includes:

1. **Classifier**: A Neural network model that maps MNIST images to class logits.
2. **Encoder**: Maps input to the representation.
3. **Latent Generator**: Produces class-specific perturbations.
4. **Decoder (Refiner)**: Converts modified latent vector back into an image.

This allows structured, smooth transitions from the original to a new class in a model-aware way

---

## Training & Usage

- There are two modes, mode 0 and mode 1. Mode 0 is for Learning and Mode 1 is for Explanation
- Download the ipynb file, set mode to 1 and run all cells except the testing part (the last cell basically)
- After that, change mode to 1, and re-run all cells
- You will be able to see the epoch-xx.png file getting uploaded in whatever address location you provided
  
Dependencies
Install required packages via:

```
pip install -r requirements.txt
#Tested on Python 3.10+ with CUDA 11. CPU is supported but significantly slower.
```
