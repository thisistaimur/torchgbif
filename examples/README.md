# TorchGBIF Examples

This directory contains example scripts demonstrating how to use TorchGBIF for various biodiversity data science tasks.

## Prerequisites

Before running the examples, you need:

1. **GBIF Account**: Register at [gbif.org](https://www.gbif.org/user/profile)
2. **Authentication**: Set environment variables:
   ```bash
   export GBIF_USERNAME="your_username"
   export GBIF_PASSWORD="your_password"
   export GBIF_EMAIL="your_email@example.com"
   ```
3. **Dependencies**: Install TorchGBIF and dependencies:
   ```bash
   pip install -e .[fair]  # Include FAIR support
   ```

## Examples Overview

### 1. Basic Usage (`basic_usage.py`)
**Perfect for**: Getting started with TorchGBIF

**What it demonstrates:**

- Setting up GBIF authentication
- Creating datasets with Hydra configuration
- Creating data loaders and iterating through batches
- Saving batches to `.pt` files

**Run it:**

```bash
python basic_usage.py
```

**Key features:**

- ✅ Hydra configuration management
- ✅ Dataset creation and validation
- ✅ Batch processing and saving
- ✅ Simple API usage

### 2. Unified Species Approach (`unified_species_approach.py`) 🆕
**Perfect for**: Single and multiple species downloads

**What it demonstrates:**

- Single interface for single and multiple species
- Backward compatibility with existing code
- Flexible parameter handling (mix single values + lists)
- Configuration-based workflows
- FAIR workflow integration
- Migration from old approaches

**Run it:**

```bash
python unified_species_approach.py
```

**Key features:**

- ✅ Single and multiple species with same API
- ✅ Automatic target column detection
- ✅ Mixed parameter usage flexibility
- ✅ FAIR workflow integration
- ✅ Migration guidance from old approaches

### 3. Species Distribution Modeling (`species_distribution_modeling.py`)
**Perfect for**: Advanced ML workflows

**What it demonstrates:**

- Using `GBIFSpeciesDataset` for specific taxa
- Advanced Hydra configuration with overrides
- Custom SQL queries for targeted data collection
- Integration with PyTorch models
- Enhanced batch saving with model predictions

**Run it:**

```bash
python species_distribution_modeling.py
```

**Key features:**

- 🐦 Species-specific dataset creation
- 🎯 Supervised learning setup
- 🤖 PyTorch model integration
- 💾 Enhanced batch saving with predictions
- 📊 Dataset analysis and statistics

### 4. FAIR Data Management (`fair_data_management.py`)
**Perfect for**: Research reproducibility

**What it demonstrates:**

- FAIR (Findable, Accessible, Interoperable, Reusable) principles
- RO-Crate packaging for research workflows
- Rich metadata and provenance tracking
- Complete research workflow examples

**Run it:**

```bash
python fair_data_management.py
```

**Key features:**

- 📦 Automatic RO-Crate generation
- 🔍 Rich metadata for discoverability
- ♻️ Reproducible research workflows
- 📊 Complete research project packaging

## Quick Start Examples

### Single Species (Traditional)
```python
from torchgbif import GBIFSpeciesDataset

# Single species download
dataset = GBIFSpeciesDataset(
    taxon_key=1340251,  # Specific bee species
    country_code="US",
    data_dir="./data/single_bee"
)
```

### Multiple Species (New Unified Approach)
```python
# Multiple families
dataset = GBIFSpeciesDataset(
    families=['Apidae', 'Megachilidae', 'Halictidae'],  # Bee families
    country_codes=['US', 'CA'],
    data_dir="./data/multi_bees"
)

# Multiple specific species
dataset = GBIFSpeciesDataset(
    scientific_names=['Apis mellifera', 'Bombus terrestris'],
    data_dir="./data/specific_bees"
)
```

### Configuration-Based
```python
from torchgbif import create_gbif_dataset

# Using configuration files
dataset = create_gbif_dataset(
    config_name="gbif_species",
    overrides=[
        "families=['Papilionidae', 'Pieridae']",  # Butterflies
        "country_codes=['US', 'CA']",
        "max_records=10000"
    ]
)
```

### FAIR-Enabled Workflows
```python
from torchgbif import create_gbif_dataloader

# Create FAIR-enabled dataloader
dataloader = create_gbif_dataloader(
    dataset=dataset,
    batch_size=64,
    save_batches=True,
    # FAIR parameters
    enable_fair=True,
    creator_name="Your Name",
    creator_email="your.email@org.edu",
    project_name="biodiversity_study"
)

# Automatically creates RO-Crates with metadata
fair_workflow = dataloader.create_fair_workflow(
    research_question="How do environmental factors affect species distributions?",
    methodology="Machine learning analysis of GBIF occurrence data"
)
```

## Configuration Examples

### Using Different Configurations
```bash
# Use species dataset configuration
python basic_usage.py dataset=gbif_species

# Override specific parameters
python basic_usage.py dataset.max_records=5000 dataloader.batch_size=64
```

### Environment Setup
Create a `.env` file in the project root:
```bash
GBIF_USERNAME=your_username
GBIF_PASSWORD=your_password
GBIF_EMAIL=your_email@example.com
```

## Output Structure

Running the examples will create:
```
examples/
├── data/                    # Downloaded GBIF data
│   ├── single_bee/         # Single species data
│   ├── multi_bees/         # Multi-species data
│   └── trees/              # Tree genera data
├── saved_batches/          # PyTorch training batches
├── fair_data/              # FAIR RO-Crates and metadata
├── outputs/                # Hydra output logs
└── *.pt files              # Individual batch examples
```

## Troubleshooting

### Authentication Issues
```
❌ Missing GBIF authentication environment variables
```
**Solution**: Set the required environment variables or check your GBIF credentials.

### Download Timeouts
```
TimeoutError: Download timed out after 3600 seconds
```
**Solution**: Increase `download_timeout` in configuration or reduce `max_records`.

### Memory Issues
```
RuntimeError: out of memory
```
**Solution**: Reduce `batch_size` or `max_records`, or enable `cache_processed=false`.

### FAIR Dependencies
```
ImportError: No module named 'rocrate'
```
**Solution**: Install FAIR support with `pip install rocrate` or `pip install -e .[fair]`.

## Next Steps

1. **Start with `basic_usage.py`** to learn the fundamentals
2. **Try `unified_species_approach.py`** to see flexible species handling
3. **Explore `species_distribution_modeling.py`** for ML integration
4. **Use `fair_data_management.py`** for research reproducibility
5. **Customize configurations** in `torchgbif/configs/`
6. **Integrate with your ML models** using the batch data

## Additional Resources

- [GBIF API Documentation](https://www.gbif.org/developer/summary)
- [PyGBIF Documentation](https://pygbif.readthedocs.io/)
- [PyTorch DataLoader Guide](https://pytorch.org/tutorials/beginner/data_loading_tutorial.html)
- [Hydra Configuration Framework](https://hydra.cc/)
- [RO-Crate Specification](https://www.researchobject.org/ro-crate/)
- [FAIR Data Principles](https://doi.org/10.1038/sdata.2016.18)
