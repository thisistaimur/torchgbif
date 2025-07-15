"""
GBIF PyTorch Dataset implementations using pygbif SQL downloads.
"""

import os
import pickle
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import requests

from pygbif import occurrences
from omegaconf import DictConfig

from .utils import ensure_dir, download_with_progress, validate_sql_query


class GBIFOccurrenceDataset(Dataset):
    """
    PyTorch Dataset for GBIF occurrence data using SQL downloads.

    This dataset downloads GBIF occurrence data using pygbif's download_sql
    functionality and provides PyTorch-compatible access to the data.

    Args:
        sql_query (str): SQL query to download GBIF data
        data_dir (str): Directory to store downloaded data
        username (str): GBIF username for authentication
        password (str): GBIF password for authentication
        email (str): Email for GBIF download requests
        feature_columns (List[str]): Columns to use as features
        target_column (str, optional): Column to use as target/label
        transform (callable, optional): Transform to apply to features
        target_transform (callable, optional): Transform to apply to targets
        cache_processed (bool): Whether to cache processed data
        max_records (int, optional): Maximum number of records to use
        download_timeout (int): Timeout for download requests in seconds
        chunk_size (int): Size of chunks for processing large datasets
    """

    def __init__(
        self,
        sql_query: str,
        data_dir: str,
        username: str,
        password: str,
        email: str,
        feature_columns: List[str],
        target_column: Optional[str] = None,
        transform: Optional[callable] = None,
        target_transform: Optional[callable] = None,
        cache_processed: bool = True,
        max_records: Optional[int] = None,
        download_timeout: int = 3600,
        chunk_size: int = 10000,
    ):
        self.sql_query = sql_query
        self.data_dir = Path(data_dir)
        self.username = username
        self.password = password
        self.email = email
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.transform = transform
        self.target_transform = target_transform
        self.cache_processed = cache_processed
        self.max_records = max_records
        self.download_timeout = download_timeout
        self.chunk_size = chunk_size

        # Ensure data directory exists
        ensure_dir(self.data_dir)

        # Validate SQL query
        validate_sql_query(sql_query)

        # Initialize data
        self.data = None
        self.targets = None
        self._download_key = None

        # Load or download data
        self._prepare_data()

    def _prepare_data(self):
        """Download and prepare the dataset."""
        cache_file = self.data_dir / "processed_data.pkl"

        if self.cache_processed and cache_file.exists():
            print("Loading cached processed data...")
            self._load_cached_data(cache_file)
        else:
            print("Downloading and processing GBIF data...")
            self._download_and_process()
            if self.cache_processed:
                self._save_cached_data(cache_file)

    def _download_and_process(self):
        """Download data from GBIF and process it."""
        # Submit download request
        print("Submitting GBIF download request...")
        download_key = occurrences.download_sql(
            sql=self.sql_query, user=self.username, pwd=self.password, email=self.email
        )

        self._download_key = download_key
        print(f"Download submitted with key: {download_key}")

        # Wait for download to complete and get download info
        print("Waiting for download to complete...")
        download_info = self._wait_for_download(download_key)

        # Download the file
        download_url = download_info["downloadLink"]
        zip_path = self.data_dir / f"{download_key}.zip"

        print("Downloading data file...")
        download_with_progress(download_url, zip_path)

        # Extract and process data
        print("Extracting and processing data...")
        self._extract_and_process(zip_path)

        # Clean up zip file
        zip_path.unlink()

    def _wait_for_download(self, download_key: str, check_interval: int = 30) -> Dict:
        """Wait for GBIF download to complete."""
        import time

        max_wait_time = self.download_timeout
        elapsed_time = 0

        while elapsed_time < max_wait_time:
            try:
                # Get download metadata
                download_info = occurrences.download_meta(download_key)
                status = download_info.get("status", "UNKNOWN")

                print(f"Download status: {status}")

                if status == "SUCCEEDED":
                    return download_info
                elif status in ["FAILED", "CANCELLED"]:
                    raise RuntimeError(f"Download failed with status: {status}")
                elif status in ["PREPARING", "RUNNING"]:
                    print(f"Download in progress... waiting {check_interval}s")
                    time.sleep(check_interval)
                    elapsed_time += check_interval
                else:
                    print(f"Unknown status: {status}, continuing to wait...")
                    time.sleep(check_interval)
                    elapsed_time += check_interval

            except Exception as e:
                print(f"Error checking download status: {e}")
                time.sleep(check_interval)
                elapsed_time += check_interval

        raise TimeoutError(f"Download timed out after {max_wait_time} seconds")

    def _extract_and_process(self, zip_path: Path):
        """Extract zip file and process the occurrence data."""
        extract_dir = self.data_dir / "extracted"
        ensure_dir(extract_dir)

        # Extract zip file
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(extract_dir)

        # Find the occurrence file (try both .txt and .csv extensions)
        occurrence_file = None

        # First try to find occurrence.txt
        for file_path in extract_dir.rglob("occurrence.txt"):
            occurrence_file = file_path
            break

        # If not found, try to find .csv files
        if occurrence_file is None:
            for file_path in extract_dir.rglob("*.csv"):
                occurrence_file = file_path
                break

        if occurrence_file is None:
            raise FileNotFoundError(
                "Could not find occurrence.txt or .csv file in downloaded data"
            )

        # Read and process data
        print(f"Reading occurrence data from {occurrence_file.name}...")
        df = pd.read_csv(occurrence_file, sep="\t", low_memory=False)

        # Apply max_records limit
        if self.max_records and len(df) > self.max_records:
            df = df.head(self.max_records)
            print(f"Limited dataset to {self.max_records} records")

        # Filter and prepare features
        self._prepare_features_and_targets(df)

        # Clean up extracted files
        import shutil

        shutil.rmtree(extract_dir)

    def _prepare_features_and_targets(self, df: pd.DataFrame):
        """Prepare feature and target tensors from the dataframe."""
        # Check if all feature columns exist
        missing_cols = [col for col in self.feature_columns if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing feature columns: {missing_cols}")

        # Prepare features
        feature_data = df[self.feature_columns].copy()

        # Handle missing values and convert to numeric where possible
        feature_data = self._preprocess_features(feature_data)

        # Convert to tensor
        self.data = torch.FloatTensor(feature_data.values)

        # Prepare targets if specified
        if self.target_column:
            if self.target_column not in df.columns:
                raise ValueError(f"Target column '{self.target_column}' not found")

            target_data = df[self.target_column].copy()
            target_data = self._preprocess_targets(target_data)
            self.targets = torch.FloatTensor(target_data.values)

        print(
            f"Prepared dataset with {len(self.data)} samples and {self.data.shape[1]} features"
        )

    def _preprocess_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Preprocess feature columns."""
        processed_df = df.copy()

        for col in processed_df.columns:
            # Try to convert to numeric
            if processed_df[col].dtype == "object":
                # Try numeric conversion
                numeric_series = pd.to_numeric(processed_df[col], errors="coerce")
                if not numeric_series.isna().all():
                    processed_df[col] = numeric_series
                else:
                    # For categorical data, use label encoding
                    processed_df[col] = pd.Categorical(processed_df[col]).codes

            # Fill missing values with mean for numeric, mode for categorical
            if processed_df[col].isna().any():
                if pd.api.types.is_numeric_dtype(processed_df[col]):
                    processed_df[col].fillna(processed_df[col].mean(), inplace=True)
                else:
                    processed_df[col].fillna(
                        processed_df[col].mode().iloc[0], inplace=True
                    )

        return processed_df

    def _preprocess_targets(self, series: pd.Series) -> pd.Series:
        """Preprocess target column."""
        if series.dtype == "object":
            # Label encode categorical targets
            return pd.Categorical(series).codes
        else:
            # Fill missing numeric targets
            return series.fillna(series.mean())

    def _save_cached_data(self, cache_file: Path):
        """Save processed data to cache."""
        cache_data = {
            "data": self.data,
            "targets": self.targets,
            "feature_columns": self.feature_columns,
            "target_column": self.target_column,
            "download_key": self._download_key,
        }

        with open(cache_file, "wb") as f:
            pickle.dump(cache_data, f)

        print(f"Cached processed data to {cache_file}")

    def _load_cached_data(self, cache_file: Path):
        """Load processed data from cache."""
        with open(cache_file, "rb") as f:
            cache_data = pickle.load(f)

        self.data = cache_data["data"]
        self.targets = cache_data["targets"]
        self._download_key = cache_data.get("download_key")

        print(f"Loaded cached data with {len(self.data)} samples")

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.data)

    def __getitem__(
        self, idx: int
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Get a sample from the dataset."""
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx]

        if self.transform:
            sample = self.transform(sample)

        if self.targets is not None:
            target = self.targets[idx]
            if self.target_transform:
                target = self.target_transform(target)
            return sample, target

        return sample

    def save_batch(self, indices: List[int], output_path: str):
        """
        Save a batch of samples to a .pt file.

        Args:
            indices: List of sample indices to include in the batch
            output_path: Path to save the batch file
        """
        batch_data = self.data[indices]

        if self.targets is not None:
            batch_targets = self.targets[indices]
            batch = {"data": batch_data, "targets": batch_targets, "indices": indices}
        else:
            batch = {"data": batch_data, "indices": indices}

        torch.save(batch, output_path)
        print(f"Saved batch of {len(indices)} samples to {output_path}")

    def get_feature_names(self) -> List[str]:
        """Get the list of feature column names."""
        return self.feature_columns

    def get_target_name(self) -> Optional[str]:
        """Get the target column name."""
        return self.target_column

    def get_download_key(self) -> Optional[str]:
        """Get the GBIF download key used for this dataset."""
        return self._download_key


class GBIFSpeciesDataset(GBIFOccurrenceDataset):
    """
    Unified dataset for GBIF species occurrence data supporting single or multiple species.

    This dataset provides flexible access to GBIF data, handling both single species
    queries and multi-species downloads with the same interface.

    Args:
        # Single species parameters (backward compatible)
        taxon_key: Optional[Union[int, List[int]]] = None,
        scientific_name: Optional[Union[str, List[str]]] = None,

        # Multi-species parameters
        taxon_keys: Optional[List[int]] = None,
        scientific_names: Optional[List[str]] = None,
        families: Optional[List[str]] = None,
        genera: Optional[List[str]] = None,
        species: Optional[List[str]] = None,

        # Geographic filters
        country_code: Optional[Union[str, List[str]]] = None,
        country_codes: Optional[List[str]] = None,

        # Other filters
        year_range: Optional[Tuple[int, int]] = None,
        has_coordinate: bool = True,
        has_geospatial_issues: bool = False,
        max_records_per_taxon: Optional[int] = None,
        **kwargs,
    """

    def __init__(
        self,
        # Unified parameters (support both single and multiple)
        taxon_key: Optional[Union[int, List[int]]] = None,
        scientific_name: Optional[Union[str, List[str]]] = None,
        country_code: Optional[Union[str, List[str]]] = None,
        # Multi-species parameters
        taxon_keys: Optional[List[int]] = None,
        scientific_names: Optional[List[str]] = None,
        families: Optional[List[str]] = None,
        genera: Optional[List[str]] = None,
        species: Optional[List[str]] = None,
        country_codes: Optional[List[str]] = None,
        # Other parameters
        year_range: Optional[Tuple[int, int]] = None,
        has_coordinate: bool = True,
        has_geospatial_issues: bool = False,
        max_records_per_taxon: Optional[int] = None,
        **kwargs,
    ):
        # Normalize inputs to lists for unified handling
        self.taxon_keys = self._normalize_to_list(taxon_key, taxon_keys)
        self.scientific_names = self._normalize_to_list(
            scientific_name, scientific_names
        )
        self.families = families or []
        self.genera = genera or []
        self.species = species or []
        self.country_codes = self._normalize_to_list(country_code, country_codes)

        # Store other parameters
        self.year_range = year_range
        self.has_coordinate = has_coordinate
        self.has_geospatial_issues = has_geospatial_issues
        self.max_records_per_taxon = max_records_per_taxon

        # Validate that at least one taxonomic criterion is provided
        self._validate_parameters()

        # Build SQL query based on parameters
        sql_query = self._build_unified_species_query()

        # Default feature columns for species data
        default_features = [
            "decimallatitude",
            "decimallongitude",
            "elevation",
            "depth",
            "year",
            "month",
            "coordinateuncertaintyinmeters",
        ]

        if "feature_columns" not in kwargs:
            kwargs["feature_columns"] = default_features

        # Auto-determine target column if not specified
        if "target_column" not in kwargs:
            kwargs["target_column"] = self._determine_target_column()

        super().__init__(sql_query=sql_query, **kwargs)

    def _normalize_to_list(self, single_value, list_value):
        """Convert single values or lists to unified list format."""
        if list_value:
            return list_value
        elif single_value is not None:
            if isinstance(single_value, list):
                return single_value
            else:
                return [single_value]
        else:
            return []

    def _validate_parameters(self):
        """Validate that at least one taxonomic criterion is provided."""
        taxonomic_criteria = [
            self.taxon_keys,
            self.scientific_names,
            self.families,
            self.genera,
            self.species,
        ]

        if not any(taxonomic_criteria):
            raise ValueError(
                "At least one taxonomic criterion must be provided: "
                "taxon_key(s), scientific_name(s), families, genera, or species"
            )

    def _determine_target_column(self):
        """Automatically determine appropriate target column."""
        if self.families:
            return "family"
        elif self.genera:
            return "genus"
        elif self.species:
            return "species"
        elif len(self.scientific_names) > 1:
            return "scientificName"
        elif len(self.taxon_keys) > 1:
            return "taxonKey"
        else:
            return "scientificName"  # Default for single species

    def _build_unified_species_query(self) -> str:
        """Build unified SQL query for single or multiple species."""

        # Base SELECT clause with comprehensive columns
        query = """
        SELECT 
            gbifId, taxonKey, scientificName, kingdom, phylum, class, 
            family, genus, species, countryCode,
            decimalLatitude, decimalLongitude, elevation, depth,
            year, month, coordinateUncertaintyInMeters,
            basisOfRecord, institutionCode, collectionCode, datasetKey
        FROM occurrence
        """

        conditions = []

        # Taxonomic conditions
        taxonomic_conditions = []

        if self.taxon_keys:
            if len(self.taxon_keys) == 1:
                taxonomic_conditions.append(f"taxonKey = {self.taxon_keys[0]}")
            else:
                keys_str = ", ".join(map(str, self.taxon_keys))
                taxonomic_conditions.append(f"taxonKey IN ({keys_str})")

        if self.scientific_names:
            if len(self.scientific_names) == 1:
                taxonomic_conditions.append(
                    f"scientificName = '{self.scientific_names[0]}'"
                )
            else:
                names_str = "', '".join(self.scientific_names)
                taxonomic_conditions.append(f"scientificName IN ('{names_str}')")

        if self.families:
            if len(self.families) == 1:
                taxonomic_conditions.append(f"family = '{self.families[0]}'")
            else:
                families_str = "', '".join(self.families)
                taxonomic_conditions.append(f"family IN ('{families_str}')")

        if self.genera:
            if len(self.genera) == 1:
                taxonomic_conditions.append(f"genus = '{self.genera[0]}'")
            else:
                genera_str = "', '".join(self.genera)
                taxonomic_conditions.append(f"genus IN ('{genera_str}')")

        if self.species:
            if len(self.species) == 1:
                taxonomic_conditions.append(f"species = '{self.species[0]}'")
            else:
                species_str = "', '".join(self.species)
                taxonomic_conditions.append(f"species IN ('{species_str}')")

        # Combine taxonomic conditions with OR (if multiple types) or AND (if same type)
        if taxonomic_conditions:
            if len(taxonomic_conditions) == 1:
                conditions.append(taxonomic_conditions[0])
            else:
                # Check if we have multiple criteria of different types - use OR
                has_multiple_types = (
                    sum(
                        [
                            bool(self.taxon_keys),
                            bool(self.scientific_names),
                            bool(self.families),
                            bool(self.genera),
                            bool(self.species),
                        ]
                    )
                    > 1
                )

                if has_multiple_types:
                    combined_taxonomic = " OR ".join(taxonomic_conditions)
                    conditions.append(f"({combined_taxonomic})")
                else:
                    # Same type, use AND (shouldn't happen with current logic, but safe)
                    conditions.extend(taxonomic_conditions)

        # Geographic conditions
        if self.country_codes:
            if len(self.country_codes) == 1:
                conditions.append(f"countryCode = '{self.country_codes[0]}'")
            else:
                countries_str = "', '".join(self.country_codes)
                conditions.append(f"countryCode IN ('{countries_str}')")

        # Temporal conditions
        if self.year_range:
            start_year, end_year = self.year_range
            conditions.append(f"year >= {start_year} AND year <= {end_year}")

        # Data quality conditions
        if self.has_coordinate:
            conditions.append(
                "decimalLatitude IS NOT NULL AND decimalLongitude IS NOT NULL"
            )

        if not self.has_geospatial_issues:
            conditions.append("hasGeospatialIssues = false")

        # Apply conditions
        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        # Add LIMIT if specified
        if self.max_records_per_taxon:
            total_taxa = (
                len(self.taxon_keys)
                + len(self.scientific_names)
                + len(self.families)
                + len(self.genera)
                + len(self.species)
            )
            if total_taxa > 0:
                total_limit = self.max_records_per_taxon * total_taxa
                query += f" LIMIT {total_limit}"

        return query

    def get_taxonomic_summary(self) -> Dict[str, Any]:
        """Get summary of taxonomic criteria used."""
        return {
            "taxon_keys": self.taxon_keys,
            "scientific_names": self.scientific_names,
            "families": self.families,
            "genera": self.genera,
            "species": self.species,
            "country_codes": self.country_codes,
            "total_taxa": (
                len(self.taxon_keys)
                + len(self.scientific_names)
                + len(self.families)
                + len(self.genera)
                + len(self.species)
            ),
            "is_multi_species": (
                len(self.taxon_keys)
                + len(self.scientific_names)
                + len(self.families)
                + len(self.genera)
                + len(self.species)
            )
            > 1,
        }
