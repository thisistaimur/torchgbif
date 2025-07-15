"""
Hydra-based configuration interface for TorchGBIF.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from .datasets import GBIFOccurrenceDataset, GBIFSpeciesDataset
from .dataloaders import GBIFDataLoader, create_dataloader_from_config


class TorchGBIFConfig:
    """
    Configuration manager for TorchGBIF using Hydra.

    This class provides a convenient interface for creating datasets and
    dataloaders from Hydra configuration files.
    """

    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize configuration manager.

        Args:
            config_dir: Path to configuration directory. If None, uses built-in configs.
        """
        if config_dir is None:
            # Use built-in configs
            config_dir = str(Path(__file__).parent / "configs")

        self.config_dir = Path(config_dir).absolute()
        self._hydra_initialized = False

    def _ensure_hydra_initialized(self):
        """Ensure Hydra is initialized with our config directory."""
        if not self._hydra_initialized:
            # Clear any existing Hydra instance
            GlobalHydra.instance().clear()

            # Initialize with our config directory
            initialize_config_dir(config_dir=str(self.config_dir), version_base=None)
            self._hydra_initialized = True

    def create_dataset(
        self,
        config_name: str = "config",
        overrides: Optional[List[str]] = None,
        **kwargs,
    ) -> GBIFOccurrenceDataset:
        """
        Create a dataset from Hydra configuration.

        Args:
            config_name: Name of the configuration file to use
            overrides: List of configuration overrides
            **kwargs: Additional parameters to override in the dataset config

        Returns:
            Configured GBIF dataset instance
        """
        self._ensure_hydra_initialized()

        # Prepare overrides
        if overrides is None:
            overrides = []

        # Add kwargs as overrides
        for key, value in kwargs.items():
            if isinstance(value, str):
                overrides.append(f"dataset.{key}='{value}'")
            else:
                overrides.append(f"dataset.{key}={value}")

        # Compose configuration
        cfg = compose(config_name=config_name, overrides=overrides)

        # Validate required authentication
        self._validate_auth_config(cfg.dataset)

        # Instantiate dataset
        dataset = hydra.utils.instantiate(cfg.dataset)

        return dataset

    def create_dataloader(
        self,
        dataset: GBIFOccurrenceDataset,
        config_name: str = "config",
        dataloader_config: str = "default",
        overrides: Optional[List[str]] = None,
        **kwargs,
    ) -> GBIFDataLoader:
        """
        Create a dataloader from Hydra configuration.

        Args:
            dataset: Dataset to create dataloader for
            config_name: Name of the main configuration file
            dataloader_config: Name of the dataloader configuration
            overrides: List of configuration overrides
            **kwargs: Additional parameters to override in the dataloader config

        Returns:
            Configured GBIF dataloader instance
        """
        self._ensure_hydra_initialized()

        # Prepare overrides
        if overrides is None:
            overrides = []

        overrides.append(f"dataloader={dataloader_config}")

        # Add kwargs as overrides
        for key, value in kwargs.items():
            if isinstance(value, str):
                overrides.append(f"dataloader.{key}='{value}'")
            else:
                overrides.append(f"dataloader.{key}={value}")

        # Compose configuration
        cfg = compose(config_name=config_name, overrides=overrides)

        # Create dataloader
        dataloader = create_dataloader_from_config(cfg.dataloader, dataset)

        return dataloader

    def create_dataset_and_dataloader(
        self,
        config_name: str = "config",
        dataset_config: str = "gbif_occurrence",
        dataloader_config: str = "default",
        overrides: Optional[List[str]] = None,
        **kwargs,
    ) -> tuple[GBIFOccurrenceDataset, GBIFDataLoader]:
        """
        Create both dataset and dataloader from configuration.

        Args:
            config_name: Name of the main configuration file
            dataset_config: Name of the dataset configuration
            dataloader_config: Name of the dataloader configuration
            overrides: List of configuration overrides
            **kwargs: Additional parameters to override

        Returns:
            Tuple of (dataset, dataloader)
        """
        if overrides is None:
            overrides = []

        overrides.append(f"dataset={dataset_config}")

        # Create dataset
        dataset = self.create_dataset(
            config_name=config_name,
            overrides=overrides,
            **{
                k: v
                for k, v in kwargs.items()
                if k.startswith("dataset_") or k in ["username", "password", "email"]
            },
        )

        # Create dataloader
        dataloader = self.create_dataloader(
            dataset=dataset,
            config_name=config_name,
            dataloader_config=dataloader_config,
            overrides=overrides,
            **{
                k: v
                for k, v in kwargs.items()
                if k.startswith("dataloader_")
                or k in ["batch_size", "shuffle", "num_workers"]
            },
        )

        return dataset, dataloader

    def _validate_auth_config(self, dataset_config: DictConfig):
        """Validate that authentication is properly configured."""
        required_fields = ["username", "password", "email"]
        missing_fields = []

        for field in required_fields:
            value = dataset_config.get(field, "")
            if not value or value == "":
                missing_fields.append(field)

        if missing_fields:
            raise ValueError(
                f"Missing GBIF authentication: {', '.join(missing_fields)}. "
                f"Please set environment variables GBIF_USERNAME, GBIF_PASSWORD, GBIF_EMAIL "
                f"or provide them in the configuration."
            )

    def list_available_configs(self) -> Dict[str, List[str]]:
        """
        List all available configuration files.

        Returns:
            Dictionary with configuration types and available options
        """
        configs = {"datasets": [], "dataloaders": [], "main": []}

        # Find dataset configs
        dataset_dir = self.config_dir / "dataset"
        if dataset_dir.exists():
            configs["datasets"] = [f.stem for f in dataset_dir.glob("*.yaml")]

        # Find dataloader configs
        dataloader_dir = self.config_dir / "dataloader"
        if dataloader_dir.exists():
            configs["dataloaders"] = [f.stem for f in dataloader_dir.glob("*.yaml")]

        # Find main configs
        configs["main"] = [f.stem for f in self.config_dir.glob("*.yaml")]

        return configs

    def get_config_template(self, config_type: str = "dataset") -> str:
        """
        Get a template for creating new configurations.

        Args:
            config_type: Type of config template ('dataset', 'dataloader')

        Returns:
            YAML template string
        """
        if config_type == "dataset":
            return """
# Custom GBIF Dataset Configuration
_target_: torchgbif.datasets.GBIFOccurrenceDataset

# Authentication
username: ${oc.env:GBIF_USERNAME}
password: ${oc.env:GBIF_PASSWORD}
email: ${oc.env:GBIF_EMAIL}

# Data storage
data_dir: "./data/custom_gbif"
cache_processed: true

# SQL query
sql_query: |
  SELECT gbifId, taxonKey, scientificName,
         decimalLatitude, decimalLongitude, elevation,
         year, month, day
  FROM occurrence 
  WHERE hasGeospatialIssues = false 
    AND hasCoordinate = true
  LIMIT 10000

# Features and targets
feature_columns:
  - decimalLatitude
  - decimalLongitude
  - elevation
  - year
  - month
  - day

target_column: null

# Processing options
max_records: 10000
download_timeout: 3600
chunk_size: 5000
"""

        elif config_type == "dataloader":
            return """
# Custom DataLoader Configuration
batch_size: 64
shuffle: true
num_workers: 2
pin_memory: false
drop_last: false
save_dir: "./data/custom_batches"
save_metadata: true
"""

        else:
            raise ValueError(f"Unknown config type: {config_type}")


# Convenience functions for quick setup
def create_gbif_dataset(
    sql_query: Optional[str] = None, config_name: str = "gbif_occurrence", **kwargs
) -> GBIFOccurrenceDataset:
    """
    Convenience function to create a GBIF dataset with minimal configuration.

    Args:
        sql_query: Custom SQL query (optional)
        config_name: Configuration preset to use
        **kwargs: Additional configuration overrides

    Returns:
        Configured GBIF dataset
    """
    config_manager = TorchGBIFConfig()

    if sql_query:
        kwargs["sql_query"] = sql_query

    return config_manager.create_dataset(
        config_name="config", overrides=[f"dataset={config_name}"], **kwargs
    )


def create_gbif_dataloader(
    dataset: GBIFOccurrenceDataset,
    batch_size: int = 32,
    save_batches: bool = True,
    **kwargs,
) -> GBIFDataLoader:
    """
    Convenience function to create a GBIF dataloader.

    Args:
        dataset: GBIF dataset instance
        batch_size: Batch size for the dataloader
        save_batches: Whether to enable batch saving
        **kwargs: Additional dataloader configuration

    Returns:
        Configured GBIF dataloader
    """
    config_manager = TorchGBIFConfig()

    dataloader_kwargs = {"batch_size": batch_size, **kwargs}
    if not save_batches:
        dataloader_kwargs["save_dir"] = None

    return config_manager.create_dataloader(dataset=dataset, **dataloader_kwargs)
