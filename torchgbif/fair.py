"""
FAIR Data Management with RO-Crates for TorchGBIF.

This module implements FAIR (Findable, Accessible, Interoperable, Reusable)
data principles by creating RO-Crates that package datasets, batches, and
metadata together for reproducible research.

FAIR Principles Implementation:
- Findable: Rich metadata with DOIs, persistent identifiers
- Accessible: Standardized access through RO-Crate format
- Interoperable: Schema.org vocabulary, standard formats
- Reusable: Clear licensing, provenance, usage instructions
"""

import json
import os
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from uuid import uuid4
import shutil

import torch
from rocrate.rocrate import ROCrate
from rocrate.model.person import Person
from rocrate.model.entity import Entity
from rocrate.model.file import File
from rocrate.model.dataset import Dataset as RODataset

from .utils import format_file_size


class FAIRDataManager:
    """
    Manages FAIR data principles implementation for TorchGBIF datasets and batches.

    Creates RO-Crates that bundle:
    - Raw and processed datasets
    - Training batches
    - Metadata and provenance
    - Configuration files
    - Documentation
    """

    def __init__(
        self,
        base_dir: Union[str, Path],
        creator_name: str,
        creator_email: str,
        creator_orcid: Optional[str] = None,
        organization: Optional[str] = None,
        license_url: str = "https://creativecommons.org/licenses/by/4.0/",
    ):
        """
        Initialize FAIR data manager.

        Args:
            base_dir: Base directory for RO-Crates
            creator_name: Name of the data creator/researcher
            creator_email: Email of the data creator
            creator_orcid: ORCID identifier (recommended)
            organization: Institution or organization
            license_url: License URL for the data
        """
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

        self.creator_name = creator_name
        self.creator_email = creator_email
        self.creator_orcid = creator_orcid
        self.organization = organization
        self.license_url = license_url

        # Generate unique identifier for this research session
        self.session_id = str(uuid4())

    def create_dataset_crate(
        self,
        dataset,
        crate_name: str,
        description: str,
        keywords: List[str],
        research_question: Optional[str] = None,
        funding_info: Optional[Dict] = None,
        additional_files: Optional[List[Path]] = None,
    ) -> Path:
        """
        Create an RO-Crate for a GBIF dataset.

        Args:
            dataset: TorchGBIF dataset instance
            crate_name: Name for the crate
            description: Description of the dataset
            keywords: Keywords for discoverability
            research_question: Research question being addressed
            funding_info: Funding information dictionary
            additional_files: Additional files to include

        Returns:
            Path to the created RO-Crate directory
        """
        crate_dir = (
            self.base_dir / f"{crate_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        crate_dir.mkdir(parents=True, exist_ok=True)

        # Create RO-Crate
        crate = ROCrate()

        # Set root dataset metadata
        crate.root_dataset.update(
            {
                "name": crate_name,
                "description": description,
                "keywords": keywords,
                "dateCreated": datetime.now(timezone.utc).isoformat(),
                "identifier": f"torchgbif-dataset-{self.session_id}",
                "license": {"@id": self.license_url},
                "creator": self._create_creator_entity(crate),
            }
        )

        if research_question:
            crate.root_dataset["about"] = research_question

        if funding_info:
            crate.root_dataset["funding"] = funding_info

        # Add dataset metadata
        dataset_metadata = self._extract_dataset_metadata(dataset)
        dataset_file = crate_dir / "dataset_metadata.json"

        with open(dataset_file, "w") as f:
            json.dump(dataset_metadata, f, indent=2, default=str)

        crate.add_file(
            dataset_file,
            properties={
                "name": "Dataset Metadata",
                "description": "Complete metadata about the GBIF dataset",
                "encodingFormat": "application/json",
                "about": "Dataset configuration and provenance",
            },
        )

        # Add dataset configuration if available
        if hasattr(dataset, "sql_query"):
            sql_file = crate_dir / "gbif_query.sql"
            with open(sql_file, "w") as f:
                f.write(dataset.sql_query)

            crate.add_file(
                sql_file,
                properties={
                    "name": "GBIF SQL Query",
                    "description": "SQL query used to download data from GBIF",
                    "encodingFormat": "application/sql",
                    "programmingLanguage": "SQL",
                },
            )

        # Add feature documentation
        features_doc = self._create_features_documentation(dataset)
        features_file = crate_dir / "features_documentation.md"
        with open(features_file, "w") as f:
            f.write(features_doc)

        crate.add_file(
            features_file,
            properties={
                "name": "Features Documentation",
                "description": "Documentation of dataset features and their meanings",
                "encodingFormat": "text/markdown",
            },
        )

        # Add GBIF citation and attribution
        gbif_attribution = self._create_gbif_attribution(dataset)
        attribution_file = crate_dir / "gbif_attribution.json"
        with open(attribution_file, "w") as f:
            json.dump(gbif_attribution, f, indent=2)

        crate.add_file(
            attribution_file,
            properties={
                "name": "GBIF Data Attribution",
                "description": "Attribution and citation information for GBIF data",
                "encodingFormat": "application/json",
            },
        )

        # Add usage instructions
        usage_doc = self._create_usage_documentation(dataset, crate_name)
        usage_file = crate_dir / "README.md"
        with open(usage_file, "w") as f:
            f.write(usage_doc)

        crate.add_file(
            usage_file,
            properties={
                "name": "Usage Instructions",
                "description": "Instructions for using this dataset",
                "encodingFormat": "text/markdown",
            },
        )

        # Add additional files if provided
        if additional_files:
            for file_path in additional_files:
                if file_path.exists():
                    dest_path = crate_dir / file_path.name
                    shutil.copy2(file_path, dest_path)
                    crate.add_file(dest_path)

        # Write RO-Crate metadata
        crate.write(crate_dir)

        print(f"📦 Created FAIR dataset RO-Crate: {crate_dir}")
        return crate_dir

    def create_batch_crate(
        self,
        batch_dir: Path,
        dataset_crate_dir: Optional[Path] = None,
        model_info: Optional[Dict] = None,
        training_config: Optional[Dict] = None,
        performance_metrics: Optional[Dict] = None,
    ) -> Path:
        """
        Create an RO-Crate for training batches.

        Args:
            batch_dir: Directory containing batch files
            dataset_crate_dir: Reference to dataset RO-Crate
            model_info: Information about the model being trained
            training_config: Training configuration
            performance_metrics: Model performance metrics

        Returns:
            Path to the created RO-Crate directory
        """
        crate_name = f"training_batches_{batch_dir.name}"
        crate_dir = (
            self.base_dir / f"{crate_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        crate_dir.mkdir(parents=True, exist_ok=True)

        # Create RO-Crate
        crate = ROCrate()

        # Set root dataset metadata
        crate.root_dataset.update(
            {
                "name": f"Training Batches: {batch_dir.name}",
                "description": f"PyTorch training batches generated from GBIF data",
                "keywords": [
                    "machine-learning",
                    "pytorch",
                    "gbif",
                    "biodiversity",
                    "training-data",
                ],
                "dateCreated": datetime.now(timezone.utc).isoformat(),
                "identifier": f"torchgbif-batches-{self.session_id}",
                "license": {"@id": self.license_url},
                "creator": self._create_creator_entity(crate),
            }
        )

        # Copy batch files to crate
        batch_files_dir = crate_dir / "batches"
        batch_files_dir.mkdir()

        batch_files = list(batch_dir.glob("*.pt"))
        for batch_file in batch_files:
            dest_file = batch_files_dir / batch_file.name
            shutil.copy2(batch_file, dest_file)

            # Calculate file hash for integrity
            file_hash = self._calculate_file_hash(dest_file)
            file_size = dest_file.stat().st_size

            crate.add_file(
                dest_file,
                properties={
                    "name": f"Training Batch {batch_file.stem}",
                    "description": "PyTorch tensor batch for machine learning",
                    "encodingFormat": "application/x-pytorch",
                    "contentSize": format_file_size(file_size),
                    "sha256": file_hash,
                },
            )

        # Add batch metadata if available
        metadata_file = batch_dir / "batch_metadata.json"
        if metadata_file.exists():
            dest_metadata = crate_dir / "batch_metadata.json"
            shutil.copy2(metadata_file, dest_metadata)

            crate.add_file(
                dest_metadata,
                properties={
                    "name": "Batch Metadata",
                    "description": "Metadata about the training batches",
                    "encodingFormat": "application/json",
                },
            )

        # Add model information
        if model_info:
            model_file = crate_dir / "model_info.json"
            with open(model_file, "w") as f:
                json.dump(model_info, f, indent=2, default=str)

            crate.add_file(
                model_file,
                properties={
                    "name": "Model Information",
                    "description": "Information about the machine learning model",
                    "encodingFormat": "application/json",
                },
            )

        # Add training configuration
        if training_config:
            config_file = crate_dir / "training_config.json"
            with open(config_file, "w") as f:
                json.dump(training_config, f, indent=2, default=str)

            crate.add_file(
                config_file,
                properties={
                    "name": "Training Configuration",
                    "description": "Configuration used for model training",
                    "encodingFormat": "application/json",
                },
            )

        # Add performance metrics
        if performance_metrics:
            metrics_file = crate_dir / "performance_metrics.json"
            with open(metrics_file, "w") as f:
                json.dump(performance_metrics, f, indent=2, default=str)

            crate.add_file(
                metrics_file,
                properties={
                    "name": "Performance Metrics",
                    "description": "Model performance metrics and evaluation results",
                    "encodingFormat": "application/json",
                },
            )

        # Link to dataset crate if provided
        if dataset_crate_dir and dataset_crate_dir.exists():
            crate.root_dataset["isBasedOn"] = {
                "@type": "Dataset",
                "name": f"Source Dataset: {dataset_crate_dir.name}",
                "identifier": str(dataset_crate_dir.absolute()),
                "description": "Original dataset used to generate these training batches",
            }

        # Create batch usage documentation
        batch_usage_doc = self._create_batch_usage_documentation(
            batch_files, model_info
        )
        usage_file = crate_dir / "README.md"
        with open(usage_file, "w") as f:
            f.write(batch_usage_doc)

        crate.add_file(
            usage_file,
            properties={
                "name": "Batch Usage Instructions",
                "description": "Instructions for using these training batches",
                "encodingFormat": "text/markdown",
            },
        )

        # Write RO-Crate metadata
        crate.write(crate_dir)

        print(f"📦 Created FAIR training batches RO-Crate: {crate_dir}")
        return crate_dir

    def create_research_crate(
        self,
        project_name: str,
        research_question: str,
        methodology: str,
        dataset_crates: List[Path],
        batch_crates: List[Path],
        results_files: Optional[List[Path]] = None,
        publications: Optional[List[Dict]] = None,
    ) -> Path:
        """
        Create a comprehensive research RO-Crate linking all components.

        Args:
            project_name: Name of the research project
            research_question: Main research question
            methodology: Research methodology description
            dataset_crates: List of dataset RO-Crate directories
            batch_crates: List of batch RO-Crate directories
            results_files: Additional result files
            publications: List of related publications

        Returns:
            Path to the research RO-Crate directory
        """
        crate_dir = (
            self.base_dir
            / f"research_{project_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        crate_dir.mkdir(parents=True, exist_ok=True)

        # Create RO-Crate
        crate = ROCrate()

        # Set root dataset metadata
        crate.root_dataset.update(
            {
                "name": f"Research Project: {project_name}",
                "description": f"Complete research workflow: {research_question}",
                "keywords": [
                    "research",
                    "biodiversity",
                    "machine-learning",
                    "reproducible-science",
                ],
                "dateCreated": datetime.now(timezone.utc).isoformat(),
                "identifier": f"torchgbif-research-{self.session_id}",
                "license": {"@id": self.license_url},
                "creator": self._create_creator_entity(crate),
                "about": research_question,
                "workflowType": "Computational Workflow",
            }
        )

        # Create research documentation
        research_doc = self._create_research_documentation(
            project_name, research_question, methodology, dataset_crates, batch_crates
        )

        readme_file = crate_dir / "README.md"
        with open(readme_file, "w") as f:
            f.write(research_doc)

        crate.add_file(
            readme_file,
            properties={
                "name": "Research Documentation",
                "description": "Complete documentation of the research project",
                "encodingFormat": "text/markdown",
            },
        )

        # Link dataset and batch crates
        linked_datasets = []
        for dataset_crate in dataset_crates:
            if dataset_crate.exists():
                linked_datasets.append(
                    {
                        "@type": "Dataset",
                        "name": dataset_crate.name,
                        "identifier": str(dataset_crate.absolute()),
                        "description": "Source dataset used in this research",
                    }
                )

        linked_batches = []
        for batch_crate in batch_crates:
            if batch_crate.exists():
                linked_batches.append(
                    {
                        "@type": "Dataset",
                        "name": batch_crate.name,
                        "identifier": str(batch_crate.absolute()),
                        "description": "Training batches used in this research",
                    }
                )

        if linked_datasets:
            crate.root_dataset["isBasedOn"] = linked_datasets

        if linked_batches:
            crate.root_dataset["hasPart"] = linked_batches

        # Add publications if provided
        if publications:
            pubs_file = crate_dir / "publications.json"
            with open(pubs_file, "w") as f:
                json.dump(publications, f, indent=2)

            crate.add_file(
                pubs_file,
                properties={
                    "name": "Related Publications",
                    "description": "Publications related to this research",
                    "encodingFormat": "application/json",
                },
            )

            crate.root_dataset["citation"] = publications

        # Add result files if provided
        if results_files:
            results_dir = crate_dir / "results"
            results_dir.mkdir()

            for result_file in results_files:
                if result_file.exists():
                    dest_file = results_dir / result_file.name
                    shutil.copy2(result_file, dest_file)
                    crate.add_file(dest_file)

        # Write RO-Crate metadata
        crate.write(crate_dir)

        print(f"📦 Created FAIR research RO-Crate: {crate_dir}")
        return crate_dir

    def _create_creator_entity(self, crate: ROCrate) -> Person:
        """Create creator entity for RO-Crate."""
        creator_id = (
            self.creator_orcid if self.creator_orcid else f"mailto:{self.creator_email}"
        )

        creator = crate.add(
            Person(
                crate,
                creator_id,
                properties={
                    "name": self.creator_name,
                    "email": self.creator_email,
                    "@type": "Person",
                },
            )
        )

        if self.creator_orcid:
            creator["identifier"] = self.creator_orcid

        if self.organization:
            creator["affiliation"] = {
                "@type": "Organization",
                "name": self.organization,
            }

        return creator

    def _extract_dataset_metadata(self, dataset) -> Dict[str, Any]:
        """Extract comprehensive metadata from dataset."""
        metadata = {
            "dataset_type": type(dataset).__name__,
            "size": len(dataset),
            "creation_date": datetime.now(timezone.utc).isoformat(),
            "feature_columns": getattr(dataset, "feature_columns", []),
            "target_column": getattr(dataset, "target_column", None),
            "gbif_download_key": getattr(dataset, "_download_key", None),
        }

        if hasattr(dataset, "sql_query"):
            metadata["sql_query"] = dataset.sql_query

        if hasattr(dataset, "data_dir"):
            metadata["data_directory"] = str(dataset.data_dir)

        return metadata

    def _create_features_documentation(self, dataset) -> str:
        """Create documentation for dataset features."""
        doc = f"""# Features Documentation

## Dataset: {type(dataset).__name__}

### Feature Columns
"""

        if hasattr(dataset, "feature_columns"):
            for i, feature in enumerate(dataset.feature_columns):
                doc += f"\n{i+1}. **{feature}**: Description of {feature} feature\n"

        doc += f"""
### Target Column
- **Target**: {getattr(dataset, 'target_column', 'None (unsupervised)')}

### Data Processing
- Dataset size: {len(dataset)} samples
- Data type: PyTorch tensors
- Preprocessing: Automatic numeric conversion and missing value handling

### Usage
```python
# Load dataset
dataset = torch.load('dataset.pt')
features = dataset['features']  # Shape: [n_samples, n_features]
targets = dataset['targets']    # Shape: [n_samples] (if supervised)
```
"""
        return doc

    def _create_gbif_attribution(self, dataset) -> Dict[str, Any]:
        """Create GBIF attribution information."""
        attribution = {
            "data_source": "Global Biodiversity Information Facility (GBIF)",
            "gbif_url": "https://www.gbif.org",
            "download_key": getattr(dataset, "_download_key", None),
            "access_date": datetime.now(timezone.utc).isoformat(),
            "citation": "Please cite GBIF as the data source and include the download DOI",
            "terms_of_use": "https://www.gbif.org/terms",
        }

        if hasattr(dataset, "sql_query"):
            attribution["query_used"] = dataset.sql_query

        return attribution

    def _create_usage_documentation(self, dataset, crate_name: str) -> str:
        """Create usage documentation for the dataset."""
        return f"""# {crate_name}

## Description
This RO-Crate contains a GBIF dataset processed for machine learning with TorchGBIF.

## Dataset Information
- **Type**: {type(dataset).__name__}
- **Size**: {len(dataset)} samples
- **Features**: {len(getattr(dataset, 'feature_columns', []))} columns
- **Target**: {getattr(dataset, 'target_column', 'None (unsupervised)')}

## Usage

### Loading the Dataset
```python
import torch
from torchgbif import BatchLoader

# Load saved batches
batch_loader = BatchLoader('batches/')
for batch in batch_loader.iterate_batches():
    features, targets = batch
    # Use with your PyTorch model
```

### Integration with PyTorch
```python
from torch.utils.data import DataLoader

# Create DataLoader
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch_idx, (features, targets) in enumerate(dataloader):
    # Training loop
    outputs = model(features)
    loss = criterion(outputs, targets)
```

## Citation
Please cite both GBIF and TorchGBIF when using this dataset:

```bibtex
@misc{{gbif_data,
  title={{GBIF Occurrence Download}},
  author={{GBIF.org}},
  year={{{datetime.now().year}}},
  url={{https://www.gbif.org}},
  note={{Download key: {getattr(dataset, '_download_key', 'N/A')}}}
}}

@software{{torchgbif,
  title={{TorchGBIF: FAIR PyTorch DataLoaders and DataSets for GBIF data}},
  author={{Taimur Khan}},
  year={{{datetime.now().year}}},
  url={{https://github.com/thisistaimur/TorchGBIF}}
}}
```

## License
This data is provided under {self.license_url}

## Contact
- Creator: {self.creator_name} ({self.creator_email})
- Organization: {self.organization or 'N/A'}
"""

    def _create_batch_usage_documentation(
        self, batch_files: List[Path], model_info: Optional[Dict]
    ) -> str:
        """Create usage documentation for training batches."""
        doc = f"""# Training Batches

## Overview
This RO-Crate contains PyTorch training batches generated from GBIF biodiversity data.

## Contents
- **Number of batches**: {len(batch_files)}
- **Format**: PyTorch tensor files (.pt)
- **Total size**: {sum(f.stat().st_size for f in batch_files) / (1024**2):.1f} MB

## Model Information
"""
        if model_info:
            for key, value in model_info.items():
                doc += f"- **{key}**: {value}\n"
        else:
            doc += "- No model information provided\n"

        doc += f"""
## Usage

### Loading Individual Batches
```python
import torch

# Load a specific batch
batch = torch.load('batches/batch_000001.pt')
features, targets = batch['batch']
print(f"Batch shape: {{features.shape}}")
```

### Loading All Batches
```python
from torchgbif import BatchLoader

batch_loader = BatchLoader('batches/')
print(f"Available batches: {{len(batch_loader)}}")

# Iterate through all batches
for batch in batch_loader.iterate_batches():
    features, targets = batch
    # Process with your model
    predictions = model(features)
```

### Training Loop Example
```python
import torch.nn as nn
import torch.optim as optim

model = YourModel()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

for batch in batch_loader.iterate_batches():
    features, targets = batch
    
    optimizer.zero_grad()
    outputs = model(features)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
```

## File Integrity
Each batch file includes SHA256 checksums for integrity verification.

## Reproducibility
To reproduce these batches, use the original dataset with the same configuration provided in the metadata.
"""
        return doc

    def _create_research_documentation(
        self,
        project_name: str,
        research_question: str,
        methodology: str,
        dataset_crates: List[Path],
        batch_crates: List[Path],
    ) -> str:
        """Create comprehensive research documentation."""
        return f"""# Research Project: {project_name}

## Research Question
{research_question}

## Methodology
{methodology}

## Data Workflow

### 1. Data Collection
- **Source**: Global Biodiversity Information Facility (GBIF)
- **Datasets**: {len(dataset_crates)} dataset(s) created
- **Processing**: TorchGBIF framework for PyTorch compatibility

### 2. Data Preparation
- **Training batches**: {len(batch_crates)} batch collection(s)
- **Format**: PyTorch tensors for machine learning

### 3. Components

#### Dataset RO-Crates
{chr(10).join(f"- {crate.name}" for crate in dataset_crates)}

#### Batch RO-Crates
{chr(10).join(f"- {crate.name}" for crate in batch_crates)}

## Reproducibility

### Requirements
```bash
pip install torchgbif
```

### Environment Setup
```bash
export GBIF_USERNAME="your_username"
export GBIF_PASSWORD="your_password"
export GBIF_EMAIL="your_email@example.com"
```

### Reproduction Steps
1. Load the dataset RO-Crates to understand data collection
2. Use the batch RO-Crates for model training
3. Follow the configuration files for exact parameter reproduction

## FAIR Principles

### Findable
- ✅ Rich metadata with identifiers
- ✅ Descriptive documentation
- ✅ Keywords for discoverability

### Accessible
- ✅ Standard RO-Crate format
- ✅ Open file formats (JSON, PyTorch tensors)
- ✅ Clear access instructions

### Interoperable
- ✅ Schema.org vocabulary
- ✅ Standard metadata formats
- ✅ PyTorch ecosystem compatibility

### Reusable
- ✅ Clear licensing information
- ✅ Detailed provenance
- ✅ Usage examples and documentation

## Citation
Please cite this research package and its components when reusing.

## Contact
- **Principal Investigator**: {self.creator_name}
- **Email**: {self.creator_email}
- **Organization**: {self.organization or 'N/A'}
- **Created**: {datetime.now().strftime('%Y-%m-%d')}
"""

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA256 hash of a file."""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()


def create_fair_batch_workflow(
    dataset,
    dataloader,
    creator_name: str,
    creator_email: str,
    project_name: str,
    research_question: str,
    base_dir: str = "./fair_data",
    **kwargs,
) -> Dict[str, Path]:
    """
    Convenience function to create a complete FAIR workflow.

    Args:
        dataset: TorchGBIF dataset
        dataloader: TorchGBIF dataloader
        creator_name: Name of the researcher
        creator_email: Email of the researcher
        project_name: Name of the research project
        research_question: Main research question
        base_dir: Base directory for FAIR data
        **kwargs: Additional arguments for FAIRDataManager

    Returns:
        Dictionary with paths to created RO-Crates
    """
    fair_manager = FAIRDataManager(
        base_dir=base_dir,
        creator_name=creator_name,
        creator_email=creator_email,
        **kwargs,
    )

    # Create dataset RO-Crate
    dataset_crate = fair_manager.create_dataset_crate(
        dataset=dataset,
        crate_name=f"{project_name}_dataset",
        description=f"GBIF dataset for {research_question}",
        keywords=["gbif", "biodiversity", "pytorch", "machine-learning"],
        research_question=research_question,
    )

    # Save batches and create batch RO-Crate
    batch_dir = Path(f"./{project_name}_batches")
    saved_files = dataloader.save_all_batches(output_dir=str(batch_dir))

    batch_crate = fair_manager.create_batch_crate(
        batch_dir=batch_dir, dataset_crate_dir=dataset_crate
    )

    # Create comprehensive research RO-Crate
    research_crate = fair_manager.create_research_crate(
        project_name=project_name,
        research_question=research_question,
        methodology="Machine learning analysis of biodiversity data using TorchGBIF",
        dataset_crates=[dataset_crate],
        batch_crates=[batch_crate],
    )

    return {
        "dataset_crate": dataset_crate,
        "batch_crate": batch_crate,
        "research_crate": research_crate,
        "saved_batches": saved_files,
    }
