"""
Utility functions for TorchGBIF datasets.
"""

import re
import requests
from pathlib import Path
from typing import Union
from tqdm import tqdm


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure that a directory exists, creating it if necessary.

    Args:
        path: Directory path to create

    Returns:
        Path object for the directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_with_progress(
    url: str, output_path: Union[str, Path], chunk_size: int = 8192
):
    """
    Download a file with progress bar.

    Args:
        url: URL to download from
        output_path: Path to save the downloaded file
        chunk_size: Size of chunks to download at a time
    """
    output_path = Path(output_path)

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with open(output_path, "wb") as f, tqdm(
        desc=output_path.name,
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))


def validate_sql_query(query: str) -> None:
    """
    Validate SQL query for basic safety and structure.

    Args:
        query: SQL query string to validate

    Raises:
        ValueError: If query appears to be invalid or unsafe
    """
    query_lower = query.lower().strip()

    # Check that it's a SELECT query
    if not query_lower.startswith("select"):
        raise ValueError("Only SELECT queries are allowed")

    # Check for potentially dangerous operations
    dangerous_keywords = [
        "drop",
        "delete",
        "insert",
        "update",
        "alter",
        "create",
        "truncate",
        "grant",
        "revoke",
        "exec",
        "execute",
    ]

    for keyword in dangerous_keywords:
        if re.search(rf"\b{keyword}\b", query_lower):
            raise ValueError(f"Query contains potentially dangerous keyword: {keyword}")

    # Check that it includes FROM occurrence
    if "from occurrence" not in query_lower:
        raise ValueError("Query must include 'FROM occurrence'")

    print("SQL query validation passed")


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human readable format.

    Args:
        size_bytes: Size in bytes

    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0B"

    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024.0 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1

    return f"{size_bytes:.1f}{size_names[i]}"


def get_available_columns() -> list:
    """
    Get list of commonly available columns in GBIF occurrence data.

    Returns:
        List of column names
    """
    return [
        # Core identification
        "gbifId",
        "taxonKey",
        "scientificName",
        "kingdom",
        "phylum",
        "class",
        "order",
        "family",
        "genus",
        "species",
        "infraspecificEpithet",
        # Location
        "decimalLatitude",
        "decimalLongitude",
        "coordinateUncertaintyInMeters",
        "elevation",
        "elevationAccuracy",
        "depth",
        "depthAccuracy",
        "country",
        "countryCode",
        "stateProvince",
        "locality",
        # Time
        "year",
        "month",
        "day",
        "eventDate",
        "dateIdentified",
        # Record details
        "basisOfRecord",
        "institutionCode",
        "collectionCode",
        "datasetKey",
        "recordedBy",
        "identifiedBy",
        "license",
        # Quality flags
        "hasGeospatialIssues",
        "hasCoordinate",
        "repatriated",
        "occurrenceStatus",
        "individualCount",
        # Media
        "mediaType",
        "issues",
    ]


def get_feature_recommendations(task_type: str = "general") -> dict:
    """
    Get recommended feature columns for different types of ML tasks.

    Args:
        task_type: Type of ML task ('general', 'species_distribution',
                  'temporal', 'spatial', 'classification')

    Returns:
        Dictionary with recommended features and targets
    """
    recommendations = {
        "general": {
            "features": [
                "decimalLatitude",
                "decimalLongitude",
                "elevation",
                "year",
                "month",
                "day",
                "coordinateUncertaintyInMeters",
            ],
            "targets": ["species", "taxonKey"],
        },
        "species_distribution": {
            "features": [
                "decimalLatitude",
                "decimalLongitude",
                "elevation",
                "depth",
                "year",
                "month",
                "coordinateUncertaintyInMeters",
            ],
            "targets": ["scientificName", "taxonKey", "species"],
        },
        "temporal": {
            "features": [
                "year",
                "month",
                "day",
                "decimalLatitude",
                "decimalLongitude",
                "elevation",
                "coordinateUncertaintyInMeters",
            ],
            "targets": ["species", "individualCount"],
        },
        "spatial": {
            "features": [
                "decimalLatitude",
                "decimalLongitude",
                "elevation",
                "depth",
                "coordinateUncertaintyInMeters",
                "country",
                "stateProvince",
            ],
            "targets": ["species", "basisOfRecord"],
        },
        "classification": {
            "features": [
                "decimalLatitude",
                "decimalLongitude",
                "elevation",
                "year",
                "month",
                "day",
                "coordinateUncertaintyInMeters",
                "country",
                "basisOfRecord",
            ],
            "targets": [
                "kingdom",
                "phylum",
                "class",
                "order",
                "family",
                "genus",
                "species",
            ],
        },
    }

    return recommendations.get(task_type, recommendations["general"])
