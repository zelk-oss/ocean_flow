# Ocean Flow Matching

Ocean surface reconstruction with Bayesian flow matching

## Folder Structure

```
ocean_flow/
├── .claude/                    # Claude agents and skills
├── configs/                    # Hydra configuration files
├── data/                       # Data directory
├── scripts/                    # Entry point scripts
├── ocean_flow/ # Source code
│   ├── data/                  # Data loading modules
│   ├── modules/               # Model modules
│   ├── networks/              # Neural network architectures
│   └── pipelines/             # Processing pipelines
├── tests/                      # Test suite
├── setup.py                    # Package setup
├── requirements.txt            # Python dependencies
├── environment.yml             # Conda environment
├── CLAUDE.md                   # Instructions for Claude Code
└── readme.md                   # This file
```

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Conda or pip

### Setup

1. **Clone the repository**:
```bash
git clone <your-repo-url>
cd Ocean Flow Matching
```

2. **Create conda environment**:
```bash
conda env create -f environment.yaml
conda activate ocean_flow
```

3. **Install dependencies**:

The template uses Hydra for configuration, PyTorch Lightning for training, and
Zarr for data storage. Install required packages:

```bash
uv pip install -r requirements.txt
pip install -e .
```

4. **Verify installation**:

To verify the installation, run the test suite. All tests should pass and the
test coverage should almost 100% (if no CUDA available, some tests are skipped).
```bash
python -m pytest -q
```

## Citation

If you use this template, please cite:

```bibtex
@software{ocean_flow,
  author =Chiara Zelco,
  title =Ocean Flow Matching,
  year = 2026,
  url = {<your-repo-url>}
}
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Submit a pull request

## Contact

Chiara Zelco - chiara.zelco@ens.psl.eu

## Acknowledgments

Built with:
- [Xarray](https://docs.xarray.dev/)
- [PyTorch](https://pytorch.org/)
- [PyTorch Lightning](https://lightning.ai/)
- [Hydra](https://hydra.cc/)
- [Weights & Biases](https://wandb.ai/)

---
This project has been created with a
[Cookiecutter](https://cookiecutter.readthedocs.io/) template from
[Surrogate-template](https://github.com/cerea-daml/surrogate-template/).