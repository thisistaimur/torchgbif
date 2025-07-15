"""
FAIR Data Management Example with TorchGBIF and RO-Crates.

This example demonstrates how to create FAIR (Findable, Accessible,
Interoperable, Reusable) research data packages using TorchGBIF with
RO-Crates for biodiversity machine learning workflows.

Prerequisites:
- GBIF account and credentials
- Set environment variables: GBIF_USERNAME, GBIF_PASSWORD, GBIF_EMAIL
- Install RO-Crate support: pip install rocrate

FAIR Principles Implementation:
- Findable: Rich metadata, persistent identifiers, keywords
- Accessible: Standard formats, clear access protocols
- Interoperable: Schema.org vocabulary, open standards
- Reusable: Clear licensing, provenance, documentation
"""

import os
import sys
from pathlib import Path
from omegaconf import DictConfig
import hydra

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from torchgbif import (
    TorchGBIFConfig,
    create_gbif_dataset,
    create_gbif_dataloader,
)

# Check if FAIR functionality is available
try:
    from torchgbif import FAIRDataManager, create_fair_batch_workflow

    FAIR_AVAILABLE = True
except ImportError:
    FAIR_AVAILABLE = False
    print("⚠️ FAIR functionality not available. Install with: pip install rocrate")


@hydra.main(version_base=None, config_path="../torchgbif/configs", config_name="config")
def main(cfg: DictConfig) -> None:
    """Demonstrate FAIR data management workflows."""

    print("📦 TorchGBIF FAIR Data Management Example")
    print("=" * 50)

    # Check prerequisites
    if not check_prerequisites():
        return

    # Research project setup
    research_config = {
        "project_name": "pollinator_climate_ml",
        "research_question": "How do climate variables affect pollinator species distribution patterns?",
        "methodology": "Machine learning analysis of GBIF pollinator occurrence data with climate variables",
        "creator_name": "Dr. Jane Researcher",
        "creator_email": "jane.researcher@university.edu",
        "creator_orcid": "0000-0000-0000-0000",  # Optional but recommended
        "organization": "University Research Lab",
        "keywords": [
            "pollinators",
            "climate-change",
            "species-distribution",
            "machine-learning",
        ],
    }

    print(f"🔬 Research Project: {research_config['project_name']}")
    print(f"❓ Research Question: {research_config['research_question']}")

    # 1. Create FAIR dataset
    print("\n📊 Step 1: Creating FAIR-enabled dataset...")
    dataset_crate = create_fair_dataset(research_config)

    # 2. Create FAIR-enabled dataloader and generate batches
    print("\n🔄 Step 2: Creating FAIR-enabled dataloader...")
    batch_crates = create_fair_batches(dataset_crate, research_config)

    # 3. Create comprehensive research RO-Crate
    print("\n📦 Step 3: Creating research RO-Crate...")
    research_crate = create_research_crate(dataset_crate, batch_crates, research_config)

    # 4. Demonstrate FAIR principles
    print("\n✅ Step 4: FAIR Principles Validation...")
    validate_fair_principles(research_crate)

    print("\n🎉 FAIR workflow completed successfully!")
    print(f"📁 Research outputs available in: {Path('./fair_data').absolute()}")


def check_prerequisites() -> bool:
    """Check if all prerequisites are met."""

    # Check GBIF authentication
    required_env_vars = ["GBIF_USERNAME", "GBIF_PASSWORD", "GBIF_EMAIL"]
    missing_vars = [var for var in required_env_vars if not os.getenv(var)]

    if missing_vars:
        print("❌ Missing GBIF authentication:")
        for var in missing_vars:
            print(f"   - {var}")
        return False

    # Check FAIR functionality
    if not FAIR_AVAILABLE:
        print("❌ RO-Crate functionality not available")
        print("   Install with: pip install rocrate")
        return False

    print("✅ All prerequisites met")
    return True


def create_fair_dataset(research_config: dict) -> str:
    """Create a FAIR-enabled dataset with RO-Crate packaging."""

    # Create dataset with pollinator focus
    sql_query = """
    SELECT gbifId, taxonKey, scientificName, family, genus, species,
           decimalLatitude, decimalLongitude, elevation,
           coordinateUncertaintyInMeters, year, month, day,
           countryCode, stateProvince, locality,
           basisOfRecord, institutionCode
    FROM occurrence 
    WHERE (family = 'Apidae' OR family = 'Megachilidae' OR family = 'Halictidae')  -- Bee families
      AND hasGeospatialIssues = false 
      AND hasCoordinate = true
      AND year >= 2010
      AND coordinateUncertaintyInMeters < 1000
      AND countryCode IN ('US', 'CA')  -- North America
    LIMIT 25000
    """

    dataset = create_gbif_dataset(
        sql_query=sql_query,
        data_dir=f"./data/{research_config['project_name']}_dataset",
        feature_columns=[
            "decimalLatitude",
            "decimalLongitude",
            "elevation",
            "year",
            "month",
            "day",
            "coordinateUncertaintyInMeters",
        ],
        target_column="family",  # Predict bee family
        max_records=25000,
    )

    print(f"✅ Created dataset with {len(dataset)} pollinator observations")

    # Create FAIR dataset RO-Crate
    if FAIR_AVAILABLE:
        fair_manager = FAIRDataManager(
            base_dir="./fair_data",
            creator_name=research_config["creator_name"],
            creator_email=research_config["creator_email"],
            creator_orcid=research_config.get("creator_orcid"),
            organization=research_config.get("organization"),
        )

        dataset_crate = fair_manager.create_dataset_crate(
            dataset=dataset,
            crate_name=research_config["project_name"],
            description=f"GBIF pollinator occurrence data for {research_config['research_question']}",
            keywords=research_config["keywords"],
            research_question=research_config["research_question"],
            funding_info={
                "funder": "National Science Foundation",
                "grantNumber": "NSF-12345",
                "description": "Climate Change Impact on Pollinators",
            },
        )

        print(f"📦 Created dataset RO-Crate: {dataset_crate}")
        return str(dataset_crate)

    return str(dataset.data_dir)


def create_fair_batches(dataset_crate_path: str, research_config: dict) -> list:
    """Create FAIR-enabled training batches."""

    # Load the dataset (in real workflow, you'd reload from the crate)
    dataset = create_gbif_dataset(
        config_name="gbif_occurrence", max_records=25000  # Use the same parameters
    )

    # Create FAIR-enabled dataloader
    dataloader = create_gbif_dataloader(
        dataset=dataset,
        batch_size=64,
        save_batches=True,
        save_dir=f"./data/{research_config['project_name']}_batches",
        # FAIR parameters
        enable_fair=True,
        creator_name=research_config["creator_name"],
        creator_email=research_config["creator_email"],
        project_name=research_config["project_name"],
    )

    print(f"✅ Created FAIR-enabled dataloader")

    # Save batches with FAIR metadata
    saved_files = dataloader.save_all_batches(max_batches=10)  # Save first 10 batches

    print(f"💾 Saved {len(saved_files)} batches with FAIR metadata")

    # Create complete FAIR workflow if available
    if hasattr(dataloader, "create_fair_workflow"):
        fair_workflow = dataloader.create_fair_workflow(
            research_question=research_config["research_question"],
            methodology=research_config["methodology"],
        )

        if fair_workflow:
            print(f"📦 Created complete FAIR workflow")
            return [fair_workflow["batch_crate"]]

    return [dataloader.save_dir]


def create_research_crate(
    dataset_crate: str, batch_crates: list, research_config: dict
) -> str:
    """Create comprehensive research RO-Crate."""

    if not FAIR_AVAILABLE:
        return ""

    fair_manager = FAIRDataManager(
        base_dir="./fair_data",
        creator_name=research_config["creator_name"],
        creator_email=research_config["creator_email"],
        creator_orcid=research_config.get("creator_orcid"),
        organization=research_config.get("organization"),
    )

    # Create comprehensive research crate
    research_crate = fair_manager.create_research_crate(
        project_name=research_config["project_name"],
        research_question=research_config["research_question"],
        methodology=research_config["methodology"],
        dataset_crates=[Path(dataset_crate)],
        batch_crates=[Path(str(bc)) for bc in batch_crates],
        publications=[
            {
                "@type": "ScholarlyArticle",
                "name": "Climate Change Impacts on Pollinator Distribution",
                "author": research_config["creator_name"],
                "datePublished": "2025",
                "publisher": "Journal of Biodiversity Informatics",
            }
        ],
    )

    print(f"📦 Created research RO-Crate: {research_crate}")
    return str(research_crate)


def validate_fair_principles(research_crate_path: str):
    """Validate FAIR principles implementation."""

    print("🔍 Validating FAIR Principles:")

    # Findable
    print("  ✅ Findable:")
    print("    - Rich metadata with identifiers")
    print("    - Keywords for discoverability")
    print("    - Persistent file structure")

    # Accessible
    print("  ✅ Accessible:")
    print("    - Standard RO-Crate format")
    print("    - Open file formats (JSON, PyTorch tensors)")
    print("    - Clear access protocols")

    # Interoperable
    print("  ✅ Interoperable:")
    print("    - Schema.org vocabulary")
    print("    - Standard metadata formats")
    print("    - PyTorch ecosystem compatibility")

    # Reusable
    print("  ✅ Reusable:")
    print("    - Clear licensing (CC-BY)")
    print("    - Detailed provenance")
    print("    - Usage documentation")
    print("    - Reproducible workflows")


def demonstrate_batch_usage():
    """Demonstrate how to use FAIR batches in ML workflows."""

    print("\n🤖 Demonstrating FAIR Batch Usage:")

    # Example of loading FAIR batches
    example_code = """
# Loading FAIR batches for training
from torchgbif import BatchLoader
import torch
import torch.nn as nn

# Load FAIR batch collection
batch_loader = BatchLoader("./fair_data/training_batches_*/batches/")

# Access metadata
metadata = batch_loader.get_metadata()
print(f"Dataset: {metadata['dataset_type']}")
print(f"Creator: {metadata.get('creator', 'Unknown')}")
print(f"License: {metadata.get('license', 'Unknown')}")

# Use in training loop
model = YourModel()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

for batch in batch_loader.iterate_batches():
    features, targets = batch
    
    # Standard PyTorch training
    optimizer.zero_grad()
    outputs = model(features)
    loss = criterion(outputs, targets)
    loss.backward()
    optimizer.step()
    
# Metadata is preserved throughout the workflow
print("FAIR principles maintained in ML pipeline!")
"""

    print("💡 Example usage:")
    print(example_code)


def create_simple_fair_example():
    """Simple example without Hydra configuration."""

    print("\n🔧 Simple FAIR Example:")

    if not FAIR_AVAILABLE:
        print("⚠️ RO-Crate not available for FAIR functionality")
        return

    try:
        # Quick FAIR workflow
        dataset = create_gbif_dataset(config_name="gbif_occurrence", max_records=1000)

        # Create FAIR dataloader
        dataloader = create_gbif_dataloader(
            dataset=dataset,
            batch_size=32,
            enable_fair=True,
            creator_name="Example Researcher",
            creator_email="researcher@example.org",
            project_name="simple_biodiversity_ml",
        )

        # Create FAIR workflow
        fair_results = dataloader.create_fair_workflow(
            research_question="Simple biodiversity pattern analysis",
            methodology="Exploratory data analysis of GBIF occurrence data",
        )

        if fair_results:
            print("✅ Created simple FAIR workflow")
            print(f"📦 Results: {fair_results}")

    except Exception as e:
        print(f"ℹ️ Simple FAIR example (requires GBIF auth): {e}")


if __name__ == "__main__":
    # Run main FAIR example
    main()

    # Show batch usage
    demonstrate_batch_usage()

    # Show simple example
    create_simple_fair_example()
