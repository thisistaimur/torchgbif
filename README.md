<p align="center">
    <img src="static/logo2.png" alt="Logo" width="400">
</p>

# TorchGBIF

[![PyPI version](https://badge.fury.io/py/torchgbif.svg)](https://badge.fury.io/py/torchgbif)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://opensource.org/license/mit/)

TorchGBIF is a PyTorch library that provides FAIR (Findable, Accessible, Interoperable, Reusable) DataLoaders and DataSets for GBIF (Global Biodiversity Information Facility) data, enabling easy access to biodiversity data for machine learning tasks with research reproducibility.

GBIF is a global network and data infrastructure that provides access to data about all types of life on Earth, including species occurrence records, taxonomic information, images, audio and more.The GBIF API provides acces to this data via SQL queries, however the data itself is not in a format that can be used directly with PyTorch. TorchGBIF provides a set of DataLoaders and DataSets that can be used to easily access GBIF data in a format that is compatible with PyTorch, while automatically generating RO-Crates for FAIR research workflows.

Read the BioHackArxiv paper for more details: [paper.md](paper.md)

## Installation

Install TorchGBIF with basic functionality:

```bash
pip install torchgbif
```

Install with FAIR data support (recommended for research):

```bash
pip install torchgbif[fair]
# or
pip install torchgbif rocrate
```

Or clone and install manually:

```bash
git clone https://github.com/thisistaimur/TorchGBIF.git
cd TorchGBIF
pip install -e .[fair]
```

## Available DataSets

- `GBIFOccurrenceDataset`: A generic dataset for GBIF occurrence data using SQL queries.
- `GBIFSpeciesDataset`: **Unified** dataset supporting both single and multiple species downloads.

## Available DataLoaders

- `GBIFDataLoader`: Extended PyTorch DataLoader with batch saving capabilities.
- `BatchLoader`: Utility for loading saved batch files.

## Quick Start

```python
import os
from torchgbif import create_gbif_dataset, create_gbif_dataloader

# Set up GBIF authentication
os.environ['GBIF_USERNAME'] = 'your_username'
os.environ['GBIF_PASSWORD'] = 'your_password' 
os.environ['GBIF_EMAIL'] = 'your_email@example.com'

# Create dataset
dataset = create_gbif_dataset(
    config_name="gbif_occurrence",
    max_records=10000
)

# Create dataloader with batch saving
dataloader = create_gbif_dataloader(
    dataset=dataset,
    batch_size=32,
    save_batches=True
)

# Process and save batches
for batch_idx, batch in enumerate(dataloader):
    features, targets = batch  # if supervised learning
    # Your PyTorch model training here
    if batch_idx >= 10:  # Save first 10 batches
        break

# Save all batches for later use
saved_files = dataloader.save_all_batches("./training_batches")
```

## Multi-Species Downloads

Unified approach supporting both single and multiple species:

```python
from torchgbif import GBIFSpeciesDataset

# Single species (backward compatible)
dataset = GBIFSpeciesDataset(
    taxon_key=1340251,  # Single bee species
    country_code="US",
    data_dir="./data/single_bee"
)

# Multiple families
dataset = GBIFSpeciesDataset(
    families=['Papilionidae', 'Pieridae', 'Lycaenidae'],  # Butterfly families
    country_codes=['US', 'CA'], 
    data_dir="./data/butterflies"
)

# Multiple genera  
dataset = GBIFSpeciesDataset(
    genera=['Quercus', 'Acer', 'Pinus'],  # Tree genera
    year_range=[2010, 2024],
    data_dir="./data/trees"
)

# Multiple specific species
dataset = GBIFSpeciesDataset(
    scientific_names=['Apis mellifera', 'Bombus terrestris'],
    data_dir="./data/specific_bees"
)

# Or use configuration (works for single or multiple)
dataset = create_gbif_dataset(
    config_name="gbif_species",
    overrides=["families=['Apidae', 'Megachilidae']"]
)
```

## Features

- **SQL-based data access**: Use pygbif's `download_sql` for flexible data queries
- **Unified species interface**: Single class handles both single and multiple species seamlessly
- **Hydra configuration**: Parameterize datasets and dataloaders with YAML configs
- **Batch saving**: Save batches as `.pt` files for reproducible ML workflows
- **FAIR data principles**: Automatic RO-Crate generation for research reproducibility
- **Backward compatibility**: All existing single-species code continues to work
- **PyTorch integration**: Native compatibility with PyTorch training loops
- **Metadata preservation**: Rich provenance tracking and documentation
- **Caching**: Automatic caching of processed data for faster subsequent runs

## FAIR Data Management

TorchGBIF implements FAIR (Findable, Accessible, Interoperable, Reusable) data principles:

```python
# Create FAIR-enabled dataloader
dataloader = create_gbif_dataloader(
    dataset=dataset,
    batch_size=32,
    save_batches=True,
    # FAIR parameters
    enable_fair=True,
    creator_name="Dr. Jane Researcher",
    creator_email="jane@university.edu",
    project_name="pollinator_climate_study"
)

# Automatically creates RO-Crates with metadata
saved_batches = dataloader.save_all_batches()

# Create complete research workflow
fair_workflow = dataloader.create_fair_workflow(
    research_question="How does climate change affect pollinator distributions?",
    methodology="Machine learning analysis of GBIF occurrence data"
)
```

For detailed FAIR usage, see [FAIR Data Management Guide](docs/fair_data_management.md).

## Configuration with Hydra

TorchGBIF uses Hydra for flexible configuration management:

```python
from torchgbif import TorchGBIFConfig

config_manager = TorchGBIFConfig()

# Create dataset with different configurations
dataset = config_manager.create_dataset(
    config_name="config",
    overrides=[
        "dataset=gbif_species",
        "dataset.taxon_key=212",     # Birds
        "dataset.country_code=US",   # United States
        "dataset.max_records=50000"
    ]
)
```

Configuration files are located in `torchgbif/configs/` and can be customized for your specific needs.

## Examples

See the [`examples/`](examples/) directory for detailed usage examples:

- **Basic Usage**: Simple dataset creation and batch processing (`basic_usage.py`)
- **Unified Species Approach**: Single interface for single or multiple species (`unified_species_approach.py`)
- **Species Distribution Modeling**: Advanced workflows for ecological research (`species_distribution_modeling.py`)
- **Custom SQL Queries**: Flexible data selection with custom SQL
- **FAIR Workflows**: Research reproducibility with RO-Crates (`fair_data_management.py`)

## Documentation

- [**FAIR Data Management**](docs/fair_data_management.md): Complete guide to FAIR research workflows
- [**Tutorial**](TUTORIAL.md): Comprehensive guide to using TorchGBIF
- [**Examples**](examples/README.md): Detailed example scripts including FAIR workflows
- [**API Reference**](https://github.com/thisistaimur/TorchGBIF): Full API documentation

## Semantic versioning

This project uses [semantic versioning](https://semver.org/). The version number is in the format `MAJOR.MINOR.PATCH`, where:

- `MAJOR` version is incremented for incompatible API changes.
- `MINOR` version is incremented for new features that are backward-compatible.
- `PATCH` version is incremented for backward-compatible bug fixes.

Releases are automatically created when tags are pushed. See [.github/RELEASE_PROCESS.md](.github/RELEASE_PROCESS.md) for details on the automated release system.




## Cite as

If you use TorchGBIF in your research, please cite it as follows:

```bibtex
@misc{torchgbif,
  author = {Taimur Khan},
  title = {TorchGBIF: FAIR PyTorch DataLoaders and DataSets for GBIF data},
  year = {2025},
  publisher = {GitHub},
  journal = {GitHub repository},
  url = {https://github.com/thisistaimur/TorchGBIF},
  note = {Implements FAIR data principles with RO-Crate packaging for biodiversity machine learning}
}
```





