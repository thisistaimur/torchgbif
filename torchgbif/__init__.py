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

# Import main classes when they are implemented
# from .datasets import GBIFSpeciesDataSet, GBIFImageDataSet, GBIFAudioDataSet
# from .dataloaders import GBIFDataLoader

__all__ = [
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    # "GBIFSpeciesDataSet",
    # "GBIFImageDataSet",
    # "GBIFAudioDataSet",
    # "GBIFDataLoader",
]
