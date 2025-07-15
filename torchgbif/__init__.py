"""
TorchGBIF: FAIR PyTorch DataLoaders and DataSets for GBIF data

This package provides PyTorch-compatible datasets and data loaders for accessing
GBIF (Global Biodiversity Information Facility) data, enabling easy integration
of biodiversity data into machine learning workflows.
"""

try:
    from ._version import version as __version__
except ImportError:
    # Fallback version if setuptools_scm is not available
    __version__ = "unknown"

__author__ = "Taimur Khan"
__email__ = ""
__license__ = "MIT"

# Import main classes
from .datasets import GBIFOccurrenceDataset, GBIFSpeciesDataset
from .dataloaders import GBIFDataLoader, BatchLoader
from .config_manager import TorchGBIFConfig, create_gbif_dataset, create_gbif_dataloader
from .utils import get_available_columns, get_feature_recommendations

# FAIR data management (optional import)
try:
    from .fair import FAIRDataManager, create_fair_batch_workflow

    _FAIR_AVAILABLE = True
except ImportError:
    _FAIR_AVAILABLE = False
    FAIRDataManager = None
    create_fair_batch_workflow = None

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    "GBIFOccurrenceDataset",
    "GBIFSpeciesDataset",
    "GBIFDataLoader",
    "BatchLoader",
    "TorchGBIFConfig",
    "create_gbif_dataset",
    "create_gbif_dataloader",
    "get_available_columns",
    "get_feature_recommendations",
]

# Add FAIR components if available
if _FAIR_AVAILABLE:
    __all__.extend(["FAIRDataManager", "create_fair_batch_workflow"])
