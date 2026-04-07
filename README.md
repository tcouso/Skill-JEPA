# Skill-JEPA
## Predictive World Models over Latent Action Manifolds

Most world models drown in physical friction, attempting to simulate the future one motor torque at a time. Skill-JEPA bypasses this computational trap via temporal abstraction. By bottlenecking physical action chunks into pure latent intents using a state-conditioned Action VAE, the predictive architecture leaps across time in macro-steps. Instead of unrolling a standard autoregressive transition $P(z_{t+1} | z_t, a_t)$ for $K$ discrete frames, Skill-JEPA evaluates a single jump: $P(\hat{z}_{t+K} | z_t, w)$, where $w$ is the compressed skill.

## Architectural Core
* **The Vision Manifold:** Standardized state embedding via headless ViT backbones ($z_t$).
* **The Action Bottleneck:** A Transformer-based VAE that compresses $K$-step physical trajectories into an isotropic Gaussian skill space.
* **The Temporal Leap:** A jumpy JEPA predictor that maps the physical anchor and the latent intent directly to the future state manifold.
* **The Stabilizer:** Latent collapse is prevented using a single-weight Sketch Isotropic Gaussian Regularizer (SIGReg), eliminating the need for complex multi-term variance/covariance penalties.

---

## Setup & Installation

Skill-JEPA requires Python 3.10 or higher.

**1. Clone the repository**
```bash
git clone [https://github.com/your-username/skill-jepa.git](https://github.com/your-username/skill-jepa.git)
cd skill-jepa
```

2. Install core dependencies
Install the local package in editable mode. This handles the architectural graph, including PyTorch, Lightning, and Einops.

```bash
pip install -e .
```

3. Mount the Data Backend & Solvers
To leverage the high-bandwidth HDF5 memory-mapped trajectory slicing and the inference planners (like the Cross-Entropy Method), install the required stable-worldmodel dependencies directly from source:

```bash
pip install git+https://github.com/galilai-group/stable-pretraining.git
pip install git+https://github.com/galilai-group/stable-worldmodel.git
```

## Training and evaluation

```bash
python src/train.py predictor.mode=jumpy training.batch_size=2
python src/eval.py ++ckpt_path=/path/to/your/model.ckpt
```
