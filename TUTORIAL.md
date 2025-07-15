# TorchGBIF Tutorial: Building PyTorch Datasets from GBIF Data

This tutorial demonstrates how to use TorchGBIF to create PyTorch-compatible datasets from GBIF (Global Biodiversity Information Facility) occurrence data using SQL queries and Hydra configuration.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Authentication Setup](#authentication-setup)
3. [Basic Dataset Creation](#basic-dataset-creation)
4. [Advanced Features](#advanced-features)
5. [Batch Processing](#batch-processing)
6. [Configuration Management](#configuration-management)
7. [Best Practices](#best-practices)

## Quick Start

```python
import os
from torchgbif import create_gbif_dataset, create_gbif_dataloader

# Set up authentication
os.environ['GBIF_USERNAME'] = 'your_username'
os.environ['GBIF_PASSWORD'] = 'your_password'
os.environ['GBIF_EMAIL'] = 'your_email@example.com'

# Create dataset
dataset = create_gbif_dataset(
    config_name="gbif_occurrence",
    max_records=10000
)

# Create dataloader
dataloader = create_gbif_dataloader(
    dataset=dataset,
    batch_size=32,
    save_batches=True
)

# Process batches
for batch_idx, batch in enumerate(dataloader):
    features, targets = batch  # if supervised
    print(f"Batch {batch_idx}: {features.shape}")
    if batch_idx >= 2:  # Process first 3 batches
        break
```

## Authentication Setup

### Method 1: Environment Variables (Recommended)

```bash
export GBIF_USERNAME="your_username"
export GBIF_PASSWORD="your_password"  
export GBIF_EMAIL="your_email@example.com"
```

### Method 2: .env File

Create a `.env` file in your project root:

```env
GBIF_USERNAME=your_username
GBIF_PASSWORD=your_password
GBIF_EMAIL=your_email@example.com
```

### Method 3: Direct Configuration

```python
from torchgbif import TorchGBIFConfig

config_manager = TorchGBIFConfig()
dataset = config_manager.create_dataset(
    username="your_username",
    password="your_password", 
    email="your_email@example.com"
)
```

## Basic Dataset Creation

### Using Built-in Configurations

```python
from torchgbif import TorchGBIFConfig

config_manager = TorchGBIFConfig()

# General occurrence data
dataset = config_manager.create_dataset(
    config_name="config",
    overrides=["dataset=gbif_occurrence"]
)

# Species-specific data
dataset = config_manager.create_dataset(
    config_name="config", 
    overrides=[
        "dataset=gbif_species",
        "dataset.taxon_key=212",  # Birds
        "dataset.country_code=US"
    ]
)
```

### Custom SQL Queries

```python
from torchgbif import create_gbif_dataset

# Custom query for plant species
plant_query = """
SELECT gbifId, taxonKey, scientificName, family, genus, species,
       decimalLatitude, decimalLongitude, elevation,
       year, month, day, countryCode
FROM occurrence 
WHERE kingdom = 'Plantae'
  AND hasGeospatialIssues = false
  AND hasCoordinate = true
  AND year >= 2015
  AND coordinateUncertaintyInMeters < 1000
LIMIT 50000
"""

dataset = create_gbif_dataset(
    sql_query=plant_query,
    feature_columns=[
        'decimalLatitude', 'decimalLongitude', 'elevation',
        'year', 'month', 'day'
    ],
    target_column='family',  # Predict plant family
    data_dir="./plant_data"
)
```

## Advanced Features

### Species Distribution Modeling

```python
from torchgbif.datasets import GBIFSpeciesDataset

# Create dataset for a specific species
dataset = GBIFSpeciesDataset(
    taxon_key=2435099,  # Aves (birds)
    country_code="CA",   # Canada
    year_range=(2010, 2024),
    has_coordinate=True,
    feature_columns=[
        'decimalLatitude', 'decimalLongitude', 'elevation',
        'year', 'month', 'day', 'coordinateUncertaintyInMeters'
    ],
    target_column='species',
    data_dir="./bird_canada_data",
    username=os.getenv('GBIF_USERNAME'),
    password=os.getenv('GBIF_PASSWORD'),
    email=os.getenv('GBIF_EMAIL')
)

print(f"Dataset size: {len(dataset)}")
print(f"Features: {dataset.get_feature_names()}")
print(f"Target: {dataset.get_target_name()}")
```

### Custom Data Preprocessing

```python
import torch
from torch.utils.data import Dataset

class CustomGBIFDataset(Dataset):
    def __init__(self, base_dataset, normalize=True):
        self.base_dataset = base_dataset
        self.normalize = normalize
        
        if normalize:
            # Calculate normalization statistics
            self._calculate_stats()
    
    def _calculate_stats(self):
        # Calculate mean and std for normalization
        all_data = torch.stack([self.base_dataset[i][0] for i in range(len(self.base_dataset))])
        self.mean = all_data.mean(dim=0)
        self.std = all_data.std(dim=0) + 1e-8  # Avoid division by zero
    
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        features, target = self.base_dataset[idx]
        
        if self.normalize:
            features = (features - self.mean) / self.std
            
        return features, target

# Use custom dataset
base_dataset = create_gbif_dataset(config_name="gbif_species")
normalized_dataset = CustomGBIFDataset(base_dataset, normalize=True)
```

## Batch Processing

### Saving Batches for Later Use

```python
from torchgbif import create_gbif_dataloader

dataloader = create_gbif_dataloader(
    dataset=dataset,
    batch_size=128,
    save_dir="./training_batches",
    save_metadata=True
)

# Save all batches
saved_files = dataloader.save_all_batches(
    output_dir="./saved_batches",
    max_batches=50  # Save first 50 batches
)

print(f"Saved {len(saved_files)} batches")
```

### Loading Saved Batches

```python
from torchgbif import BatchLoader

# Load saved batches
batch_loader = BatchLoader("./saved_batches")

print(f"Available batches: {len(batch_loader)}")
print(batch_loader.summary())

# Iterate through batches
for batch in batch_loader.iterate_batches():
    features, targets = batch
    # Process with your model
    predictions = model(features)
```

### Enhanced Batch Saving with Model Predictions

```python
import torch.nn as nn

# Your PyTorch model
class SpeciesClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)

# Process and save enhanced batches
model = SpeciesClassifier(input_dim=7, num_classes=100)

for batch_idx, (features, targets) in enumerate(dataloader):
    # Get model predictions
    with torch.no_grad():
        predictions = model(features)
        probabilities = torch.softmax(predictions, dim=1)
    
    # Create enhanced batch
    enhanced_batch = {
        'features': features,
        'targets': targets,
        'predictions': predictions,
        'probabilities': probabilities,
        'batch_metadata': {
            'batch_idx': batch_idx,
            'batch_size': len(features),
            'feature_names': dataset.get_feature_names()
        }
    }
    
    # Save enhanced batch
    torch.save(enhanced_batch, f"enhanced_batch_{batch_idx:04d}.pt")
```

## Configuration Management

### Creating Custom Configurations

Create `custom_config.yaml`:

```yaml
# Custom configuration for marine species
dataset:
  _target_: torchgbif.datasets.GBIFOccurrenceDataset
  username: ${oc.env:GBIF_USERNAME}
  password: ${oc.env:GBIF_PASSWORD}
  email: ${oc.env:GBIF_EMAIL}
  
  data_dir: "./marine_data"
  
  sql_query: |
    SELECT gbifId, taxonKey, scientificName, kingdom, phylum, class,
           decimalLatitude, decimalLongitude, depth,
           year, month, day, coordinateUncertaintyInMeters
    FROM occurrence 
    WHERE kingdom = 'Animalia'
      AND depth IS NOT NULL
      AND depth > 0
      AND hasGeospatialIssues = false
      AND year >= 2010
    LIMIT 100000
  
  feature_columns:
    - decimalLatitude
    - decimalLongitude  
    - depth
    - year
    - month
    - day
    
  target_column: "class"
  max_records: 100000

dataloader:
  batch_size: 64
  shuffle: true
  save_dir: "./marine_batches"
```

### Using Custom Configurations

```python
from torchgbif import TorchGBIFConfig

config_manager = TorchGBIFConfig(config_dir="./my_configs")
dataset, dataloader = config_manager.create_dataset_and_dataloader(
    config_name="custom_config"
)
```

### Configuration Overrides

```python
# Override configuration parameters
dataset = config_manager.create_dataset(
    config_name="config",
    overrides=[
        "dataset=gbif_species",
        "dataset.max_records=25000",
        "dataset.country_code=BR",  # Brazil
        "dataset.taxon_key=359",    # Mammalia
        "dataloader.batch_size=256"
    ]
)
```

## Best Practices

### 1. Start Small and Scale Up

```python
# Start with small dataset for testing
test_dataset = create_gbif_dataset(
    config_name="gbif_occurrence",
    max_records=1000  # Small test set
)

# Scale up after validation
production_dataset = create_gbif_dataset(
    config_name="gbif_occurrence", 
    max_records=100000  # Full dataset
)
```

### 2. Efficient Data Processing

```python
# Use appropriate batch sizes
dataloader = create_gbif_dataloader(
    dataset=dataset,
    batch_size=128,     # Balance memory and efficiency
    num_workers=4,      # Parallel data loading
    pin_memory=True,    # Faster GPU transfer
    save_dir="./batches"
)
```

### 3. Error Handling and Validation

```python
try:
    dataset = create_gbif_dataset(
        sql_query=custom_query,
        max_records=50000,
        download_timeout=7200  # 2 hours for large datasets
    )
    
    # Validate dataset
    if len(dataset) == 0:
        raise ValueError("Dataset is empty - check SQL query")
        
    print(f"✅ Dataset created successfully: {len(dataset)} samples")
    
except ValueError as e:
    print(f"❌ Configuration error: {e}")
except TimeoutError as e:
    print(f"❌ Download timeout: {e}")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
```

### 4. Memory Management

```python
# For large datasets, disable caching if memory is limited
dataset = create_gbif_dataset(
    config_name="gbif_occurrence",
    cache_processed=False,  # Don't cache processed data
    max_records=None,       # No limit
    chunk_size=5000        # Process in smaller chunks
)
```

### 5. Reproducible Research

```python
# Set random seeds for reproducibility
import torch
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

# Use fixed configuration files
dataset = config_manager.create_dataset(
    config_name="research_config_v1",  # Version your configs
    overrides=["dataset.max_records=50000"]
)

# Save dataset metadata
metadata = {
    'gbif_download_key': dataset.get_download_key(),
    'feature_names': dataset.get_feature_names(),
    'target_name': dataset.get_target_name(),
    'dataset_size': len(dataset),
    'creation_date': datetime.now().isoformat()
}

with open("dataset_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)
```

## Common Use Cases

### Biodiversity Hotspot Analysis

```python
# Multi-region comparison
regions = ['US', 'BR', 'AU', 'ZA']  # Different continents

datasets = {}
for region in regions:
    datasets[region] = create_gbif_dataset(
        config_name="gbif_species",
        country_code=region,
        taxon_key=212,  # Birds
        max_records=20000,
        data_dir=f"./data_{region}"
    )

# Compare biodiversity patterns
for region, dataset in datasets.items():
    print(f"{region}: {len(dataset)} observations")
```

### Temporal Analysis

```python
# Year-by-year analysis
temporal_datasets = {}
for year in range(2015, 2025):
    sql_query = f"""
    SELECT gbifId, scientificName, decimalLatitude, decimalLongitude,
           elevation, month, day, coordinateUncertaintyInMeters
    FROM occurrence 
    WHERE year = {year}
      AND hasGeospatialIssues = false
      AND hasCoordinate = true
      AND kingdom = 'Plantae'
    LIMIT 10000
    """
    
    temporal_datasets[year] = create_gbif_dataset(
        sql_query=sql_query,
        data_dir=f"./temporal_data_{year}"
    )
```

### Model Training Pipeline

```python
# Complete ML pipeline
def create_training_pipeline(config_name="gbif_species"):
    # Create dataset
    dataset = create_gbif_dataset(config_name=config_name)
    
    # Split dataset (implement your own train/val split)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create dataloaders
    train_loader = create_gbif_dataloader(
        train_dataset, batch_size=128, shuffle=True,
        save_dir="./train_batches"
    )
    
    val_loader = create_gbif_dataloader(
        val_dataset, batch_size=128, shuffle=False,
        save_dir="./val_batches"
    )
    
    return train_loader, val_loader

# Use in training
train_loader, val_loader = create_training_pipeline()
```

This tutorial covers the main features of TorchGBIF. For more advanced usage, see the example scripts in the `examples/` directory.
