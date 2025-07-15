"""
Advanced example demonstrating species distribution modeling with TorchGBIF.

This example shows:
1. Using the GBIFSpeciesDataset for specific species
2. Custom SQL queries for targeted data collection
3. Advanced configuration with Hydra overrides
4. Batch processing for machine learning workflows
5. Integration with PyTorch models

Prerequisites:
- GBIF account and credentials
- Set environment variables: GBIF_USERNAME, GBIF_PASSWORD, GBIF_EMAIL
"""

import os
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from omegaconf import DictConfig
import hydra

# Add project root to path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from torchgbif import TorchGBIFConfig, GBIFSpeciesDataset


class SimpleSpeciesPredictor(nn.Module):
    """Simple neural network for species prediction."""

    def __init__(self, input_dim: int, num_species: int, hidden_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_species),
        )

    def forward(self, x):
        return self.network(x)


@hydra.main(version_base=None, config_path="../torchgbif/configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Advanced species distribution modeling example."""

    print("🐦 TorchGBIF Species Distribution Modeling Example")
    print("=" * 60)

    # Check authentication
    if not all(
        os.getenv(var) for var in ["GBIF_USERNAME", "GBIF_PASSWORD", "GBIF_EMAIL"]
    ):
        print("❌ Please set GBIF authentication environment variables")
        return

    print("✅ GBIF authentication configured")

    # Configuration for bird species in North America
    config_manager = TorchGBIFConfig()

    # Create dataset for birds (Aves) in North America
    print("\n📊 Creating bird species dataset...")

    overrides = [
        "dataset=gbif_species",
        "dataset.taxon_key=212",  # Aves (birds)
        "dataset.country_code=US",
        "dataset.year_range=[2015,2024]",
        "dataset.max_records=50000",
        "dataset.target_column=species",
    ]

    dataset = config_manager.create_dataset(config_name="config", overrides=overrides)

    print(f"✅ Created dataset with {len(dataset)} bird observations")
    print(f"📊 Features: {dataset.get_feature_names()}")

    # Create dataloader with larger batches for ML training
    dataloader = config_manager.create_dataloader(
        dataset=dataset,
        dataloader_config="large_batch",
        batch_size=256,
        save_dir="./bird_batches",
    )

    print(f"✅ Created dataloader with batch size: {dataloader.batch_size}")

    # Analyze the dataset
    print("\n📈 Dataset Analysis:")
    analyze_dataset(dataset, dataloader)

    # Save training batches
    print("\n💾 Saving training batches...")
    batch_dir = Path("./bird_training_batches")
    saved_files = dataloader.save_all_batches(
        output_dir=str(batch_dir), max_batches=20  # Save first 20 batches
    )

    print(f"✅ Saved {len(saved_files)} training batches")

    # Demonstrate model training preparation
    print("\n🤖 Preparing for model training...")
    prepare_model_training(dataset, dataloader)

    print("\n🎉 Advanced example completed!")


def analyze_dataset(dataset, dataloader):
    """Analyze the dataset characteristics."""

    # Get first batch for analysis
    first_batch = next(iter(dataloader))

    if isinstance(first_batch, tuple):
        features, targets = first_batch

        print(f"  📊 Feature tensor shape: {features.shape}")
        print(f"  🎯 Target tensor shape: {targets.shape}")
        print(f"  📍 Feature ranges:")

        feature_names = dataset.get_feature_names()
        for i, name in enumerate(feature_names):
            if i < features.shape[1]:
                min_val = features[:, i].min().item()
                max_val = features[:, i].max().item()
                mean_val = features[:, i].mean().item()
                print(
                    f"     {name}: {min_val:.2f} to {max_val:.2f} (mean: {mean_val:.2f})"
                )

        print(f"  🏷️ Unique species in batch: {len(torch.unique(targets))}")

    else:
        print(f"  📊 Feature tensor shape: {first_batch.shape}")


def prepare_model_training(dataset, dataloader):
    """Demonstrate preparation for PyTorch model training."""

    # Get dataset characteristics
    first_batch = next(iter(dataloader))

    if isinstance(first_batch, tuple):
        features, targets = first_batch
        input_dim = features.shape[1]
        num_species = len(torch.unique(targets))

        print(f"  📐 Input dimension: {input_dim}")
        print(f"  🏷️ Number of species: {num_species}")

        # Create a simple model
        model = SimpleSpeciesPredictor(
            input_dim=input_dim, num_species=num_species, hidden_dim=128
        )

        print(f"  🧠 Created model: {model}")
        print(f"  🔢 Model parameters: {sum(p.numel() for p in model.parameters()):,}")

        # Test forward pass
        with torch.no_grad():
            predictions = model(features)
            print(f"  ✅ Forward pass successful: {predictions.shape}")

        # Demonstrate batch saving with model predictions
        save_predictions_example(model, features, targets, dataset)

    else:
        print("  ℹ️ No targets available - unsupervised learning setup")


def save_predictions_example(model, features, targets, dataset):
    """Example of saving model predictions with batch data."""

    with torch.no_grad():
        predictions = model(features)
        probabilities = torch.softmax(predictions, dim=1)

    # Create enhanced batch with predictions
    enhanced_batch = {
        "features": features,
        "targets": targets,
        "predictions": predictions,
        "probabilities": probabilities,
        "feature_names": dataset.get_feature_names(),
        "target_name": dataset.get_target_name(),
        "model_info": {
            "model_type": "SimpleSpeciesPredictor",
            "input_dim": features.shape[1],
            "num_classes": predictions.shape[1],
        },
    }

    # Save enhanced batch
    output_path = "enhanced_batch_with_predictions.pt"
    torch.save(enhanced_batch, output_path)

    print(f"  💾 Saved enhanced batch with predictions to: {output_path}")
    print(
        f"  📊 Batch includes: features, targets, predictions, probabilities, metadata"
    )


def custom_sql_example():
    """Example with custom SQL query for specific research needs."""

    print("\n🔍 Custom SQL Query Example")
    print("=" * 40)

    # Custom query for endangered bird species with recent observations
    custom_sql = """
    SELECT 
        gbifId, taxonKey, scientificName, species,
        decimalLatitude, decimalLongitude, elevation, depth,
        coordinateUncertaintyInMeters, year, month, day,
        basisOfRecord, institutionCode, countryCode,
        stateProvince, locality, habitat
    FROM occurrence 
    WHERE taxonKey = 212  -- Aves (birds)
      AND hasGeospatialIssues = false 
      AND hasCoordinate = true
      AND year >= 2020
      AND countryCode IN ('US', 'CA', 'MX')  -- North America
      AND coordinateUncertaintyInMeters < 1000  -- High precision only
      AND basisOfRecord = 'HUMAN_OBSERVATION'  -- Reliable observations
    ORDER BY year DESC, month DESC
    LIMIT 25000
    """

    print("📝 Custom SQL Query for North American birds (2020+):")
    print(custom_sql)

    try:
        from torchgbif import create_gbif_dataset

        dataset = create_gbif_dataset(
            sql_query=custom_sql,
            data_dir="./custom_bird_data",
            feature_columns=[
                "decimalLatitude",
                "decimalLongitude",
                "elevation",
                "coordinateUncertaintyInMeters",
                "year",
                "month",
                "day",
            ],
            target_column="species",
            max_records=25000,
        )

        print(f"✅ Custom dataset would contain {len(dataset)} observations")

    except Exception as e:
        print(f"ℹ️ Custom dataset creation example (would require GBIF auth): {e}")


if __name__ == "__main__":
    # Run main example
    main()

    # Show custom SQL example
    custom_sql_example()
