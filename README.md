# Explanation Selection in Abductive Learning

Research code for *Explanation Selection in Abductive Learning* by En-Hao Gao and Yuan Jiang. The experiments study which concept labels training selects when several assignments satisfy the same logical constraints, and when later updates preserve those labels.

## Quick start

Use Python 3.10. The reference runs used PyTorch 2.4.1 and torchvision 0.19.1.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python reproduce.py check
python reproduce.py figures
```

Both commands work on CPU with the files in this repository. The first checks the numerical updates and the reported comparisons. The second recreates the four main figures and the two appendix figures in `generated/figures/`.

## Experiments

Run an experiment with `python reproduce.py <command>`. The table follows the order of the paper.

| Paper | What the experiment compares | Command |
| --- | --- | --- |
| Sections 6.1–6.2, Fig. 2 | Four Gram orientations from the same initial head, followed by metric changes after capture | `selection` |
| Section 6.2, App. G.4–G.5 | Capture timing across step sizes and capture with a rank-deficient shared head | `capture_timing`, `restricted` |
| Section 6.3, App. G.6 | Return to a wrong stationary state after perturbation, and recovery from random initial heads | `minima`, `recovery` |
| Section 6.3, Fig. 3, App. G.7 | How often random heads reach a wrong state as direct supervision weakens, and whether reweighting the queries restores the correct labels | `natural_access` |
| Section 6.4, Fig. 4, App. G.8 | End-to-end SGD and Adam from identical initial networks, paired with a true-label reference | `joint_training` |
| App. C.11 | Local stability on finite candidate sets | `local` |
| App. G.9 | Concept recovery under Sudoku rules as fewer boards reveal a cell | `sudoku` |

`protocols/` specifies the data splits, seeds, optimizer settings and stopping criteria. `results/` contains the corresponding reference measurements. Experiment outputs go to `generated/` by default. To repeat an experiment, give it a fresh `--output` directory.

For a short CPU run, use `python reproduce.py local --quick`. The frozen-feature experiments use the supplied ResNet-18 feature cache. Full `natural_access`, `joint_training` and `sudoku` runs require CUDA. For the two experiments that train the encoder, download CIFAR-10 first. Torchvision loads the pretrained weights on first use.

```bash
python -c "from torchvision.datasets import CIFAR10; CIFAR10('generated/cifar10', train=True, download=True); CIFAR10('generated/cifar10', train=False, download=True)"
python reproduce.py joint_training --device cuda:0 --data-root generated/cifar10
```

The default joint-training rates reproduce Figure 4. Use `--rates original` with `joint_training` to reproduce the original-grid comparison in Appendix G.8. Both settings use the same four evaluation sets, initial heads and minibatch order. Appendix G.8 describes how the rates were chosen on a separate tuning set.

## Data

The feature cache contains 10,000 training and 2,000 test feature vectors from the official bird and cat classes, with their original split indices. To regenerate it, run the following command after downloading CIFAR-10. The artwork in Figure 2 includes the 24 example images and the Lucide icons credited in the SVG. [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) · [ResNet-18 weights](https://docs.pytorch.org/vision/0.19/models/generated/torchvision.models.resnet18.html) · [Icon attribution](assets/truth_icons.svg)

```bash
python verification/prepare_cifar_llp_features.py --data-root generated/cifar10 \
  --output generated/features.npz --metadata generated/feature_metadata.json --device cpu
```

The authors' code is released under the [MIT License](LICENSE). Third-party data, model weights and icons retain their upstream terms.
