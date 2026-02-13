"""
Data loading module.
Handles reading CSV files and preparing data for evaluation.
"""
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict
from config import REQUIRED_COLUMNS


class MTDataset:
    """
    Class to handle MT evaluation dataset loading and preparation.
    """
    
    def __init__(self, filepath: Path):
        """
        Initialize dataset from CSV file.
        
        Args:
            filepath: Path to the CSV file
        """
        self.filepath = filepath
        self.df = pd.read_csv(filepath)
        self._validate_columns()
        
    def _validate_columns(self) -> None:
        """Validate that required columns exist in the dataset."""
        missing_cols = [col for col in REQUIRED_COLUMNS if col not in self.df.columns]
        if missing_cols:
            raise ValueError(
                f"Missing required columns in {self.filepath}: {missing_cols}"
            )
    
    def get_mt_systems(self) -> List[str]:
        """
        Get list of MT system column names (excluding Source and Reference).
        
        Returns:
            List of MT system names
        """
        return [col for col in self.df.columns if col not in REQUIRED_COLUMNS]
    
    def prepare_data(self, mt_system: str = None) -> Tuple[List[str], List[str], List[str]]:
        """
        Prepare data for evaluation, handling missing values.
        
        Args:
            mt_system: Name of MT system column. If None, only returns sources and references.
        
        Returns:
            Tuple of (sources, references, hypotheses) if mt_system provided,
            or (sources, references, []) if mt_system is None
        """
        # Clean data: remove rows with NaN in Source or Reference
        df_clean = self.df.dropna(subset=REQUIRED_COLUMNS)
        
        if mt_system is None:
            sources = df_clean['Source'].astype(str).tolist()
            references = df_clean['Reference'].astype(str).tolist()
            return sources, references, []
        
        # Further remove rows with NaN in the MT system column
        df_mt = df_clean.dropna(subset=[mt_system])
        
        if len(df_mt) == 0:
            raise ValueError(f"No valid translations found for MT system: {mt_system}")
        
        sources = df_mt['Source'].astype(str).tolist()
        references = df_mt['Reference'].astype(str).tolist()
        hypotheses = df_mt[mt_system].astype(str).tolist()
        
        return sources, references, hypotheses
    
    def get_dataset_info(self) -> Dict:
        """
        Get information about the dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        return {
            'total_rows': len(self.df),
            'valid_rows': len(self.df.dropna(subset=REQUIRED_COLUMNS)),
            'mt_systems': self.get_mt_systems(),
            'num_mt_systems': len(self.get_mt_systems())
        }


def load_datasets(language_files: Dict[str, Path]) -> Dict[str, MTDataset]:
    """
    Load multiple datasets from file paths.
    
    Args:
        language_files: Dictionary mapping language names to file paths
    
    Returns:
        Dictionary mapping language names to MTDataset objects
    """
    datasets = {}
    for language, filepath in language_files.items():
        try:
            datasets[language] = MTDataset(filepath)
        except FileNotFoundError:
            print(f"Warning: File not found for {language}: {filepath}")
        except Exception as e:
            print(f"Error loading {language} dataset: {e}")
    
    return datasets
