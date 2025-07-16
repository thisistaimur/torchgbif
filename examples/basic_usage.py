"""
Basic example of using TorchGBIF with Hydra configuration.

This example demonstrates:
1. Setting up GBIF authentication
2. Creating a dataset with Hydra configuration
3. Creating a dataloader
4. Iterating through batches
5. Saving batches to .pt files

Prerequisites:
- GBIF account and credentials
- Set environment variables: GBIF_USERNAME, GBIF_PASSWORD, GBIF_EMAIL
"""

import os
from pathlib import Path
import torch
from omegaconf import DictConfig, OmegaConf
import hydra

# Add project root to path for imports
import sys

sys.path.append(str(Path(__file__).parent.parent))

from torchgbif import (
    TorchGBIFConfig,
    create_gbif_dataset,
    create_gbif_dataloader,
    get_feature_recommendations,
)


@hydra.main(version_base=None, config_path="../torchgbif/configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main example function using Hydra configuration."""

    print("🌍 TorchGBIF Basic Example")
    print("=" * 50)

    # Print configuration
    print("\n📋 Configuration:")
    print(OmegaConf.to_yaml(cfg, resolve=True))

    # Note: Authentication is handled via configuration files
    # check_authentication()  # Disabled since we use config-based auth

    # Create dataset using Hydra config
    print("\n📊 Creating GBIF dataset...")
    dataset = hydra.utils.instantiate(cfg.dataset)

    print(f"✅ Dataset created with {len(dataset)} samples")
    print(f"📊 Features: {dataset.get_feature_names()}")
    print(f"🎯 Target: {dataset.get_target_name()}")

    # Create dataloader
    print("\n🔄 Creating dataloader...")
    config_manager = TorchGBIFConfig()
    dataloader = config_manager.create_dataloader(dataset, config_name="config")

    print(f"✅ Dataloader created with batch size: {dataloader.batch_size}")

    # Iterate through a few batches
    print("\n🔢 Processing batches...")

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= 3:  # Process only first 3 batches for demo
            break

        if isinstance(batch, tuple):
            features, targets = batch
            print(
                f"  Batch {batch_idx}: Features shape: {features.shape}, Targets shape: {targets.shape}"
            )
        else:
            features = batch
            print(f"  Batch {batch_idx}: Features shape: {features.shape}")

        # Save first batch as example
        if batch_idx == 0:
            output_path = "example_batch.pt"
            dataloader.save_batch(batch, batch_idx)
            print(f"  💾 Saved batch to: {output_path}")

    # Demonstrate batch saving
    print("\n💾 Saving all batches...")
    batch_dir = Path("./saved_batches")
    saved_files = dataloader.save_all_batches(
        output_dir=str(batch_dir)  # Removed max_batches=5 to save all batches
    )

    print(f"✅ Saved {len(saved_files)} batches to {batch_dir}")
    print(
        f"📊 With batch size {dataloader.batch_size}, this covers all {len(dataset)} samples"
    )

    # Load saved batches
    print("\n📂 Loading saved batches...")
    from torchgbif import BatchLoader

    batch_loader = BatchLoader(str(batch_dir))
    print(f"📊 {batch_loader.summary()}")

    # Load a specific batch
    first_batch = batch_loader.load_batch(0)
    print(
        f"📦 Loaded first batch with shape: {first_batch[0].shape if isinstance(first_batch, tuple) else first_batch.shape}"
    )

    print("\n🎉 Example completed successfully!")


def check_authentication():
    """Check if GBIF authentication is properly set up."""
    required_env_vars = ["GBIF_USERNAME", "GBIF_PASSWORD", "GBIF_EMAIL"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]

    if missing_vars:
        print("❌ Missing GBIF authentication environment variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\nPlease set these environment variables:")
        print("export GBIF_USERNAME='your_username'")
        print("export GBIF_PASSWORD='your_password'")
        print("export GBIF_EMAIL='your_email@example.com'")
        print("\nOr create a .env file with these variables.")
        sys.exit(1)
    else:
        print("✅ GBIF authentication configured")


def simple_example():
    """Simple example without Hydra configuration."""
    print("\n🔧 Simple API Example (without Hydra)")
    print("=" * 50)

    # Define features directly
    feature_columns = [
        "decimallatitude",
        "decimallongitude",
        "year",
        "month",
        "taxonkey",
    ]

    # Create dataset with simple API - use direct import to avoid Hydra parsing issues
    sql_query = """
    SELECT gbifid, taxonkey, scientificname,
           decimallatitude, decimallongitude,
           "year", "month"
    FROM occurrence 
    WHERE hasGeospatialIssues = false 
      AND hasCoordinate = true
      AND countryCode = 'DE'
      AND taxonKey = 2481912
    LIMIT 10000
    """

    try:
        # Use direct dataset creation instead of config-based approach
        from torchgbif.datasets import GBIFOccurrenceDataset
        from torchgbif.dataloaders import GBIFDataLoader
        import os

        dataset = GBIFOccurrenceDataset(
            sql_query=sql_query,
            data_dir="./simple_data",
            username=os.getenv("GBIF_USERNAME"),
            password=os.getenv("GBIF_PASSWORD"),
            email=os.getenv("GBIF_EMAIL"),
            feature_columns=feature_columns,
            max_records=10000,
        )

        dataloader = GBIFDataLoader(
            dataset=dataset, batch_size=128, save_dir="./batches"
        )

        print(f"✅ Created dataset with {len(dataset)} samples")
        print(f"✅ Created dataloader with batch size {dataloader.batch_size}")

        # Iterate through a few batches (following main() pattern)
        print("\n🔢 Processing batches...")

        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= 3:  # Process only first 3 batches for demo
                break

            if isinstance(batch, tuple):
                features, targets = batch
                print(
                    f"  Batch {batch_idx}: Features shape: {features.shape}, Targets shape: {targets.shape}"
                )
            else:
                features = batch
                print(f"  Batch {batch_idx}: Features shape: {features.shape}")

            # Save first batch as example
            if batch_idx == 0:
                output_path = "simple_example_batch.pt"
                dataloader.save_batch(batch, batch_idx)
                print(f"  💾 Saved batch to: {output_path}")

        # Demonstrate batch saving (following main() pattern)
        print("\n💾 Saving all batches...")
        batch_dir = Path("./simple_batches")
        saved_files = dataloader.save_all_batches(output_dir=str(batch_dir))

        print(f"✅ Saved {len(saved_files)} batches to {batch_dir}")
        print(
            f"📊 With batch size {dataloader.batch_size}, this covers all {len(dataset)} samples"
        )

        # Load saved batches (following main() pattern)
        print("\n📂 Loading saved batches...")
        from torchgbif import BatchLoader

        batch_loader = BatchLoader(str(batch_dir))
        print(f"📊 {batch_loader.summary()}")

        # Load a specific batch
        first_batch = batch_loader.load_batch(0)
        print(
            f"📦 Loaded first batch with shape: {first_batch[0].shape if isinstance(first_batch, tuple) else first_batch.shape}"
        )

        print("\n🎉 Simple example completed successfully!")

    except ValueError as e:
        print(f"❌ Error: {e}")
        print("Make sure GBIF authentication is set up!")
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("Check your GBIF credentials and network connection!")


if __name__ == "__main__":
    # Run the main Hydra example
    # main()

    # Also run the simple example
    simple_example()
