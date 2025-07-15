"""
PyTorch DataLoaders for GBIF datasets with batch saving functionality.
"""

from pathlib import Path
from typing import Optional, Callable, List, Dict, Any
import torch
from torch.utils.data import DataLoader as TorchDataLoader, Dataset
from omegaconf import DictConfig
import json
from datetime import datetime

from .utils import ensure_dir


class GBIFDataLoader(TorchDataLoader):
    """
    Extended PyTorch DataLoader for GBIF datasets with batch saving capabilities.

    This DataLoader extends the standard PyTorch DataLoader to provide
    functionality for saving batches to .pt files for later use.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 1,
        shuffle: bool = False,
        sampler: Optional[torch.utils.data.Sampler] = None,
        batch_sampler: Optional[torch.utils.data.Sampler] = None,
        num_workers: int = 0,
        collate_fn: Optional[Callable] = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: Optional[Callable] = None,
        multiprocessing_context=None,
        generator=None,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
        # Custom parameters for batch saving
        save_dir: Optional[str] = None,
        save_metadata: bool = True,
        # FAIR data management parameters
        enable_fair: bool = False,
        creator_name: Optional[str] = None,
        creator_email: Optional[str] = None,
        project_name: Optional[str] = None,
    ):
        # Prepare arguments for parent DataLoader
        dataloader_kwargs = {
            "dataset": dataset,
            "batch_size": batch_size,
            "shuffle": shuffle,
            "sampler": sampler,
            "batch_sampler": batch_sampler,
            "num_workers": num_workers,
            "collate_fn": collate_fn,
            "pin_memory": pin_memory,
            "drop_last": drop_last,
            "timeout": timeout,
            "worker_init_fn": worker_init_fn,
            "multiprocessing_context": multiprocessing_context,
            "generator": generator,
        }

        # Only add multiprocessing-related parameters if num_workers > 0
        if num_workers > 0:
            dataloader_kwargs["prefetch_factor"] = prefetch_factor
            dataloader_kwargs["persistent_workers"] = persistent_workers

        super().__init__(**dataloader_kwargs)

        self.save_dir = Path(save_dir) if save_dir else None
        self.save_metadata = save_metadata
        self._batch_counter = 0

        # FAIR data management
        self.enable_fair = enable_fair
        self.creator_name = creator_name
        self.creator_email = creator_email
        self.project_name = project_name
        self._fair_manager = None

        if self.save_dir:
            ensure_dir(self.save_dir)

        # Initialize FAIR manager if enabled
        if self.enable_fair and self.creator_name and self.creator_email:
            try:
                from .fair import FAIRDataManager

                fair_base_dir = (
                    self.save_dir.parent / "fair_data"
                    if self.save_dir
                    else "./fair_data"
                )
                self._fair_manager = FAIRDataManager(
                    base_dir=fair_base_dir,
                    creator_name=self.creator_name,
                    creator_email=self.creator_email,
                )
            except ImportError:
                print("⚠️ RO-Crate not available. Install with: pip install rocrate")
                self.enable_fair = False

    def save_batch(
        self, batch: Any, batch_idx: int, metadata: Optional[Dict] = None
    ) -> str:
        """
        Save a batch to a .pt file.

        Args:
            batch: The batch data to save
            batch_idx: Index/identifier for the batch
            metadata: Optional metadata to save with the batch

        Returns:
            Path to the saved batch file
        """
        if self.save_dir is None:
            raise ValueError("save_dir must be specified to save batches")

        filename = f"batch_{batch_idx:06d}.pt"
        filepath = self.save_dir / filename

        # Prepare batch data for saving
        batch_data = {
            "batch": batch,
            "batch_idx": batch_idx,
            "batch_size": (
                len(batch[0]) if isinstance(batch, (list, tuple)) else len(batch)
            ),
            "timestamp": datetime.now().isoformat(),
        }

        if metadata:
            batch_data["metadata"] = metadata

        if self.save_metadata and hasattr(self.dataset, "get_feature_names"):
            feature_names = self.dataset.get_feature_names()
            # Convert OmegaConf objects to regular Python types for saving
            try:
                from omegaconf import OmegaConf

                if (
                    hasattr(feature_names, "_content")
                    or str(type(feature_names)).find("omegaconf") != -1
                ):
                    feature_names = OmegaConf.to_object(feature_names)
                elif not isinstance(feature_names, (list, tuple)):
                    feature_names = list(feature_names)
            except ImportError:
                if not isinstance(feature_names, (list, tuple)):
                    feature_names = list(feature_names)
            batch_data["feature_names"] = feature_names

            target_name = getattr(self.dataset, "get_target_name", lambda: None)()
            if target_name:
                try:
                    from omegaconf import OmegaConf

                    if (
                        hasattr(target_name, "_content")
                        or str(type(target_name)).find("omegaconf") != -1
                    ):
                        target_name = OmegaConf.to_object(target_name)
                except ImportError:
                    pass
            batch_data["target_name"] = target_name

        torch.save(batch_data, filepath)
        return str(filepath)

    def save_all_batches(
        self, output_dir: Optional[str] = None, max_batches: Optional[int] = None
    ) -> List[str]:
        """
        Iterate through all batches and save them to .pt files.

        Args:
            output_dir: Directory to save batches (uses self.save_dir if None)
            max_batches: Maximum number of batches to save

        Returns:
            List of paths to saved batch files
        """
        if output_dir:
            original_save_dir = self.save_dir
            self.save_dir = Path(output_dir)
            ensure_dir(self.save_dir)

        saved_files = []

        try:
            for batch_idx, batch in enumerate(self):
                if max_batches and batch_idx >= max_batches:
                    break

                filepath = self.save_batch(batch, batch_idx)
                saved_files.append(filepath)

                if batch_idx % 10 == 0:
                    print(f"Saved {batch_idx + 1} batches...")

            print(f"Successfully saved {len(saved_files)} batches to {self.save_dir}")

            # Save batch metadata
            if self.save_metadata:
                self._save_batch_metadata(saved_files)

            # Create FAIR RO-Crate if enabled
            if self.enable_fair and self._fair_manager:
                self._create_fair_batch_crate(saved_files)

        finally:
            if output_dir:
                self.save_dir = original_save_dir

        return saved_files

    def _save_batch_metadata(self, saved_files: List[str]):
        """Save metadata about all saved batches."""
        metadata = {
            "num_batches": len(saved_files),
            "batch_size": self.batch_size,
            "total_samples": len(self.dataset),
            "dataset_type": type(self.dataset).__name__,
            "saved_files": saved_files,
            "created_at": datetime.now().isoformat(),
        }

        if hasattr(self.dataset, "get_feature_names"):
            feature_names = self.dataset.get_feature_names()
            # Convert OmegaConf/Hydra objects to regular Python types
            try:
                from omegaconf import OmegaConf

                if (
                    hasattr(feature_names, "_content")
                    or str(type(feature_names)).find("omegaconf") != -1
                ):
                    feature_names = OmegaConf.to_object(feature_names)
                elif not isinstance(feature_names, (list, tuple)):
                    feature_names = list(feature_names)
            except ImportError:
                if not isinstance(feature_names, (list, tuple)):
                    feature_names = list(feature_names)
            metadata["feature_names"] = feature_names

            target_name = getattr(self.dataset, "get_target_name", lambda: None)()
            if target_name:
                try:
                    from omegaconf import OmegaConf

                    if (
                        hasattr(target_name, "_content")
                        or str(type(target_name)).find("omegaconf") != -1
                    ):
                        target_name = OmegaConf.to_object(target_name)
                except ImportError:
                    pass
            metadata["target_name"] = target_name

        if hasattr(self.dataset, "get_download_key"):
            metadata["gbif_download_key"] = self.dataset.get_download_key()

        metadata_file = self.save_dir / "batch_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved batch metadata to {metadata_file}")

    def _create_fair_batch_crate(self, saved_files: List[str]):
        """Create FAIR RO-Crate for saved batches."""
        try:
            project_name = self.project_name or "torchgbif_batches"

            # Get model info if available from dataset
            model_info = {}
            if hasattr(self.dataset, "get_feature_names"):
                feature_names = self.dataset.get_feature_names()
                # Convert OmegaConf objects to regular Python types
                try:
                    from omegaconf import OmegaConf

                    if (
                        hasattr(feature_names, "_content")
                        or str(type(feature_names)).find("omegaconf") != -1
                    ):
                        feature_names = OmegaConf.to_object(feature_names)
                    elif not isinstance(feature_names, (list, tuple)):
                        feature_names = list(feature_names)
                except ImportError:
                    if not isinstance(feature_names, (list, tuple)):
                        feature_names = list(feature_names)
                model_info["input_features"] = feature_names
                model_info["num_features"] = len(feature_names)
            if hasattr(self.dataset, "get_target_name"):
                target_name = self.dataset.get_target_name()
                try:
                    from omegaconf import OmegaConf

                    if (
                        hasattr(target_name, "_content")
                        or str(type(target_name)).find("omegaconf") != -1
                    ):
                        target_name = OmegaConf.to_object(target_name)
                except ImportError:
                    pass
                model_info["target"] = target_name

            # Get training config
            training_config = {
                "batch_size": self.batch_size,
                "shuffle": getattr(self, "shuffle", False),
                "num_workers": getattr(self, "num_workers", 0),
                "dataset_size": len(self.dataset),
                "num_batches": len(saved_files),
            }

            # Create batch RO-Crate
            crate_dir = self._fair_manager.create_batch_crate(
                batch_dir=self.save_dir,
                model_info=model_info,
                training_config=training_config,
            )

            print(f"📦 Created FAIR RO-Crate for batches: {crate_dir}")
            return crate_dir

        except Exception as e:
            print(f"⚠️ Could not create FAIR RO-Crate: {e}")
            return None

    def create_fair_workflow(
        self,
        research_question: str,
        methodology: str = "Machine learning analysis of GBIF biodiversity data",
        additional_metadata: Optional[Dict] = None,
    ) -> Optional[Dict[str, Path]]:
        """
        Create a complete FAIR workflow with RO-Crates.

        Args:
            research_question: Main research question being addressed
            methodology: Research methodology description
            additional_metadata: Additional metadata to include

        Returns:
            Dictionary with paths to created RO-Crates
        """
        if not self.enable_fair or not self._fair_manager:
            print(
                "⚠️ FAIR workflow not enabled. Set enable_fair=True and provide creator info."
            )
            return None

        try:
            from .fair import create_fair_batch_workflow

            project_name = self.project_name or "torchgbif_research"

            # Create complete FAIR workflow
            fair_crates = create_fair_batch_workflow(
                dataset=self.dataset,
                dataloader=self,
                creator_name=self.creator_name,
                creator_email=self.creator_email,
                project_name=project_name,
                research_question=research_question,
                methodology=methodology,
                **(additional_metadata or {}),
            )

            print("🎉 Created complete FAIR research workflow!")
            return fair_crates

        except Exception as e:
            print(f"⚠️ Could not create FAIR workflow: {e}")
            return None


def create_dataloader_from_config(
    config: DictConfig, dataset: Dataset
) -> GBIFDataLoader:
    """
    Create a GBIFDataLoader from a Hydra configuration.

    Args:
        config: Hydra configuration containing dataloader parameters
        dataset: The dataset to create a dataloader for

    Returns:
        Configured GBIFDataLoader instance
    """
    # Extract dataloader parameters from config
    loader_params = {
        "batch_size": config.get("batch_size", 32),
        "shuffle": config.get("shuffle", True),
        "num_workers": config.get("num_workers", 0),
        "pin_memory": config.get("pin_memory", False),
        "drop_last": config.get("drop_last", False),
        "save_dir": config.get("save_dir", None),
        "save_metadata": config.get("save_metadata", True),
    }

    # Remove None values
    loader_params = {k: v for k, v in loader_params.items() if v is not None}

    return GBIFDataLoader(dataset=dataset, **loader_params)


class BatchLoader:
    """
    Utility class for loading saved batches from .pt files.
    """

    def __init__(self, batch_dir: str):
        """
        Initialize BatchLoader.

        Args:
            batch_dir: Directory containing saved batch files
        """
        self.batch_dir = Path(batch_dir)

        if not self.batch_dir.exists():
            raise FileNotFoundError(f"Batch directory not found: {batch_dir}")

        # Load metadata if available
        self.metadata = None
        metadata_file = self.batch_dir / "batch_metadata.json"
        if metadata_file.exists():
            with open(metadata_file, "r") as f:
                self.metadata = json.load(f)

        # Find all batch files
        self.batch_files = sorted(list(self.batch_dir.glob("batch_*.pt")))

        if not self.batch_files:
            raise FileNotFoundError(f"No batch files found in {batch_dir}")

        print(f"Found {len(self.batch_files)} batch files")

    def __len__(self) -> int:
        """Return number of saved batches."""
        return len(self.batch_files)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Load a specific batch by index."""
        if idx >= len(self.batch_files):
            raise IndexError(f"Batch index {idx} out of range")

        batch_file = self.batch_files[idx]
        return torch.load(batch_file, weights_only=False)

    def load_batch(self, batch_idx: int) -> Any:
        """Load the actual batch data (without metadata) by index."""
        batch_data = self[batch_idx]
        return batch_data["batch"]

    def get_batch_info(self, batch_idx: int) -> Dict[str, Any]:
        """Get metadata for a specific batch."""
        batch_data = self[batch_idx]
        return {k: v for k, v in batch_data.items() if k != "batch"}

    def iterate_batches(self):
        """Iterate through all saved batches."""
        for i in range(len(self)):
            yield self.load_batch(i)

    def get_metadata(self) -> Optional[Dict[str, Any]]:
        """Get overall metadata about the saved batches."""
        return self.metadata

    def summary(self) -> str:
        """Get a summary of the saved batches."""
        summary_lines = [
            f"Batch Directory: {self.batch_dir}",
            f"Number of Batches: {len(self.batch_files)}",
        ]

        if self.metadata:
            summary_lines.extend(
                [
                    f"Batch Size: {self.metadata.get('batch_size', 'Unknown')}",
                    f"Total Samples: {self.metadata.get('total_samples', 'Unknown')}",
                    f"Dataset Type: {self.metadata.get('dataset_type', 'Unknown')}",
                    f"Created: {self.metadata.get('created_at', 'Unknown')}",
                ]
            )

            if "feature_names" in self.metadata:
                features = self.metadata["feature_names"]
                summary_lines.append(
                    f"Features ({len(features)}): {', '.join(features[:5])}{'...' if len(features) > 5 else ''}"
                )

            if "target_name" in self.metadata and self.metadata["target_name"]:
                summary_lines.append(f"Target: {self.metadata['target_name']}")

        return "\n".join(summary_lines)
