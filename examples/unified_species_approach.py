"""
Unified Species Download Examples

This example demonstrates the unified approach in TorchGBIF where a single
GBIFSpeciesDataset class handles both single and multiple species seamlessly.
"""

import os
from torchgbif import GBIFSpeciesDataset, create_gbif_dataset, create_gbif_dataloader


def single_species_examples():
    """Traditional single species downloads (backward compatible)."""

    print("🐝 Single Species Examples (Backward Compatible)")
    print("-" * 50)

    # Example 1: Single species by taxon key
    print("1. Single species by taxon key:")
    dataset1 = GBIFSpeciesDataset(
        username=os.getenv("GBIF_USERNAME"),
        password=os.getenv("GBIF_PASSWORD"),
        email=os.getenv("GBIF_EMAIL"),
        taxon_key=1340251,  # Single taxon key
        country_code="US",  # Single country
        data_dir="./data/single_bee",
        max_records=1000,
    )
    print(f"   ✅ {len(dataset1)} records for single bee species")
    print(f"   📊 Summary: {dataset1.get_taxonomic_summary()}")

    # Example 2: Single species by scientific name
    print("\n2. Single species by scientific name:")
    dataset2 = GBIFSpeciesDataset(
        username=os.getenv("GBIF_USERNAME"),
        password=os.getenv("GBIF_PASSWORD"),
        email=os.getenv("GBIF_EMAIL"),
        scientific_name="Quercus alba",  # White oak
        data_dir="./data/single_oak",
        max_records=1000,
    )
    print(f"   ✅ {len(dataset2)} records for white oak")
    print(f"   📊 Summary: {dataset2.get_taxonomic_summary()}")

    return dataset1, dataset2


def multi_species_examples():
    """Multiple species downloads using the same class."""

    print("\n🦋 Multi-Species Examples (New Capabilities)")
    print("-" * 50)

    # Example 1: Multiple families
    print("1. Multiple butterfly families:")
    dataset1 = GBIFSpeciesDataset(
        username=os.getenv("GBIF_USERNAME"),
        password=os.getenv("GBIF_PASSWORD"),
        email=os.getenv("GBIF_EMAIL"),
        families=["Papilionidae", "Pieridae", "Lycaenidae"],  # Multiple families
        country_codes=["US", "CA"],  # Multiple countries
        data_dir="./data/multi_butterflies",
        max_records=5000,
    )
    print(f"   ✅ {len(dataset1)} records for butterfly families")
    print(f"   📊 Summary: {dataset1.get_taxonomic_summary()}")

    # Example 2: Multiple specific species
    print("\n2. Multiple specific bee species:")
    dataset2 = GBIFSpeciesDataset(
        username=os.getenv("GBIF_USERNAME"),
        password=os.getenv("GBIF_PASSWORD"),
        email=os.getenv("GBIF_EMAIL"),
        scientific_names=["Apis mellifera", "Bombus terrestris", "Osmia bicornis"],
        data_dir="./data/specific_bees",
        max_records=3000,
    )
    print(f"   ✅ {len(dataset2)} records for specific bee species")
    print(f"   📊 Summary: {dataset2.get_taxonomic_summary()}")

    # Example 3: Multiple genera
    print("\n3. Multiple tree genera:")
    dataset3 = GBIFSpeciesDataset(
        username=os.getenv("GBIF_USERNAME"),
        password=os.getenv("GBIF_PASSWORD"),
        email=os.getenv("GBIF_EMAIL"),
        genera=["Quercus", "Acer", "Pinus"],  # Oak, Maple, Pine
        country_codes=["US", "CA"],
        data_dir="./data/tree_genera",
        max_records=4000,
    )
    print(f"   ✅ {len(dataset3)} records for tree genera")
    print(f"   📊 Summary: {dataset3.get_taxonomic_summary()}")

    return dataset1, dataset2, dataset3


def unified_approach_examples():
    """Examples showing the flexibility of the unified approach."""

    print("\n⚡ Unified Approach Examples")
    print("-" * 50)

    # Example 1: Mix single and list parameters
    print("1. Mixed single/multiple parameters:")
    dataset1 = GBIFSpeciesDataset(
        username=os.getenv("GBIF_USERNAME"),
        password=os.getenv("GBIF_PASSWORD"),
        email=os.getenv("GBIF_EMAIL"),
        taxon_key=[1340251, 1340358],  # Can pass list to single parameter too!
        country_code="US",  # Single country
        data_dir="./data/mixed_approach",
        max_records=2000,
    )
    print(f"   ✅ {len(dataset1)} records using mixed parameters")
    print(f"   📊 Summary: {dataset1.get_taxonomic_summary()}")

    # Example 2: Configuration-based approach
    print("\n2. Configuration-based (single species):")
    dataset2 = create_gbif_dataset(
        config_name="gbif_species",
        overrides=["taxon_key=212", "country_code=US", "max_records=1500"],  # Birds
    )
    print(f"   ✅ {len(dataset2)} bird records via config")

    # Example 3: Configuration-based (multi-species)
    print("\n3. Configuration-based (multi-species):")
    dataset3 = create_gbif_dataset(
        config_name="gbif_species",
        overrides=[
            "families=['Apidae', 'Megachilidae']",  # Bee families
            "country_codes=['US', 'CA']",
            "max_records=3000",
        ],
    )
    print(f"   ✅ {len(dataset3)} bee records via config")

    return dataset1, dataset2, dataset3


def migration_examples():
    """Show how old multi-species code can be migrated."""

    print("\n🔄 Migration Examples")
    print("-" * 50)

    print("OLD approach (no longer needed):")
    print(
        """
    # This was the old way:
    from torchgbif import GBIFMultiSpeciesDataset  # ❌ No longer needed
    
    dataset = GBIFMultiSpeciesDataset(
        families=['Apidae', 'Megachilidae'],
        country_codes=['US', 'CA']
    )
    """
    )

    print("\nNEW unified approach:")
    print(
        """
    # This is the new way:
    from torchgbif import GBIFSpeciesDataset  # ✅ Unified class
    
    dataset = GBIFSpeciesDataset(
        families=['Apidae', 'Megachilidae'],  # Same parameters!
        country_codes=['US', 'CA']
    )
    """
    )

    print("✅ Migration is simple - just change the class name!")


def fair_workflow_example():
    """FAIR workflow with unified approach."""

    print("\n📦 FAIR Workflow Example")
    print("-" * 50)

    # Create multi-species dataset
    dataset = GBIFSpeciesDataset(
        username=os.getenv("GBIF_USERNAME"),
        password=os.getenv("GBIF_PASSWORD"),
        email=os.getenv("GBIF_EMAIL"),
        families=["Apidae", "Megachilidae", "Halictidae"],  # Multiple bee families
        country_codes=["US", "CA"],
        data_dir="./data/fair_bees",
        max_records=5000,
    )

    # Create FAIR-enabled dataloader
    dataloader = create_gbif_dataloader(
        dataset=dataset,
        batch_size=64,
        save_batches=True,
        # FAIR parameters
        enable_fair=True,
        creator_name="Unified Species Researcher",
        creator_email="researcher@example.org",
        project_name="unified_bee_study",
    )

    print(f"✅ Created unified FAIR dataset with {len(dataset)} bee records")
    print(f"📊 Taxonomic summary: {dataset.get_taxonomic_summary()}")

    try:
        # Create FAIR workflow
        fair_workflow = dataloader.create_fair_workflow(
            research_question="Comparative analysis of multiple bee families",
            methodology="Unified approach for single and multi-species data",
        )

        if fair_workflow:
            print("📦 Created FAIR workflow with unified approach")

    except Exception as e:
        print(f"ℹ️ FAIR workflow creation requires authentication: {e}")

    return dataset, dataloader


def show_advantages():
    """Show advantages of the unified approach."""

    print("\n🎯 Advantages of Unified Approach")
    print("=" * 50)

    advantages = [
        "🔄 **Backward Compatible**: All existing single-species code works unchanged",
        "📈 **Seamless Scaling**: Same interface for 1 species or 100 species",
        "🧹 **Cleaner Codebase**: One class instead of multiple specialized classes",
        "⚡ **Flexible Parameters**: Mix single values and lists as needed",
        "🔧 **Easy Migration**: Just change class name for multi-species",
        "📦 **FAIR Ready**: Full RO-Crate support for all configurations",
        "⚙️ **Config Driven**: Same YAML configs work for any number of species",
        "🎯 **Auto-Detection**: Automatically determines appropriate target columns",
    ]

    for advantage in advantages:
        print(f"   {advantage}")


def main():
    """Run all unified approach examples."""

    print("🌍 TorchGBIF Unified Species Download Approach")
    print("=" * 55)

    # Check authentication
    required_vars = ["GBIF_USERNAME", "GBIF_PASSWORD", "GBIF_EMAIL"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]

    if missing_vars:
        print("❌ Missing GBIF authentication variables:")
        for var in missing_vars:
            print(f"   - {var}")
        print("\n💡 Code examples and advantages still shown below...")
        show_advantages()
        migration_examples()
        return

    try:
        # Run examples
        single_datasets = single_species_examples()
        multi_datasets = multi_species_examples()
        unified_datasets = unified_approach_examples()
        fair_dataset, fair_dataloader = fair_workflow_example()

        print("\n🎉 All unified approach examples completed!")

        # Show summary
        all_datasets = (
            single_datasets + multi_datasets + unified_datasets + (fair_dataset,)
        )
        total_records = sum(len(d) for d in all_datasets)
        print(f"\n📊 Total records across all examples: {total_records}")

    except Exception as e:
        print(f"❌ Error: {e}")

    # Show advantages and migration info
    show_advantages()
    migration_examples()


if __name__ == "__main__":
    main()
