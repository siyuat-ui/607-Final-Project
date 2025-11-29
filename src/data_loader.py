"""
Data loading and validation for latent class model.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from pathlib import Path


class DataLoader:
    """
    Load and validate categorical data for latent class analysis.
    """
    
    def __init__(self, filepath: str):
        """
        Initialize DataLoader.
        
        Parameters
        ----------
        filepath : str
            Path to CSV file containing categorical data
        """
        self.filepath = Path(filepath)
        self.data = None
        self.categories = None
        self.variable_names = None
        
    def load_data(self, 
                  has_header: bool = True,
                  zero_indexed: bool = True) -> Tuple[np.ndarray, List[int], List[str]]:
        """
        Load categorical data from CSV file.
        
        Parameters
        ----------
        has_header : bool, default=True
            Whether the CSV file has a header row with variable names
        zero_indexed : bool, default=True
            Whether categories start from 0. If False, will convert to 0-indexed.
            
        Returns
        -------
        X : np.ndarray, shape (n, m)
            Data matrix where each column is a categorical variable
        categories : list of int, length m
            Number of categories for each variable
        variable_names : list of str, length m
            Names of variables
            
        Raises
        ------
        FileNotFoundError
            If the specified file does not exist
        ValueError
            If data is not valid categorical data
        """
        if not self.filepath.exists():
            raise FileNotFoundError(f"Data file not found: {self.filepath}")
        
        # Load CSV
        if has_header:
            df = pd.read_csv(self.filepath)
            self.variable_names = df.columns.tolist()
        else:
            df = pd.read_csv(self.filepath, header=None)
            self.variable_names = [f"Var_{i}" for i in range(df.shape[1])]
        
        # Convert to numpy array
        X = df.values
        
        # Check for missing values
        if np.any(pd.isna(X)):
            raise ValueError("Data contains missing values. Please handle missing data before analysis.")
        
        # Convert to integer type
        try:
            X = X.astype(int)
        except ValueError:
            raise ValueError("Data must contain only integer categorical values")
        
        # Convert to 0-indexed if needed
        if not zero_indexed:
            X = X - 1
            if np.any(X < 0):
                raise ValueError("After converting to 0-indexed, some values are negative. "
                               "Check that your data is 1-indexed.")
        
        # Validate non-negative
        if np.any(X < 0):
            raise ValueError("Data contains negative values. Categories should be non-negative integers.")
        
        # Determine number of categories for each variable
        n, m = X.shape
        self.categories = []
        for r in range(m):
            unique_vals = np.unique(X[:, r])
            max_val = np.max(unique_vals)
            min_val = np.min(unique_vals)
            
            if min_val != 0:
                raise ValueError(f"Variable {self.variable_names[r]} does not start from 0. "
                               f"Minimum value is {min_val}")
            
            # Number of categories is max_val + 1 (since 0-indexed)
            num_cats = max_val + 1
            self.categories.append(num_cats)
            
            # Check for gaps in categories
            if len(unique_vals) < num_cats:
                missing_cats = set(range(num_cats)) - set(unique_vals)
                print(f"Warning: Variable '{self.variable_names[r]}' has missing categories: {missing_cats}")
        
        self.data = X
        
        print(f"Loaded data: {n} samples, {m} variables")
        print(f"Categories per variable: {self.categories}")
        
        return X, self.categories, self.variable_names
    
    def get_summary_statistics(self) -> pd.DataFrame:
        """
        Get summary statistics for the loaded data.
        
        Returns
        -------
        pd.DataFrame
            Summary statistics including category counts and proportions
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        summaries = []
        for r, var_name in enumerate(self.variable_names):
            unique, counts = np.unique(self.data[:, r], return_counts=True)
            proportions = counts / len(self.data)
            
            summary = {
                'Variable': var_name,
                'Num_Categories': self.categories[r],
                'Most_Common': unique[np.argmax(counts)],
                'Most_Common_Prop': proportions[np.argmax(counts)]
            }
            summaries.append(summary)
        
        return pd.DataFrame(summaries)
    
    def save_processed_data(self, output_path: str) -> None:
        """
        Save the processed (validated and 0-indexed) data to CSV.
        
        Parameters
        ----------
        output_path : str
            Path where to save the processed data
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        df = pd.DataFrame(self.data, columns=self.variable_names)
        df.to_csv(output_path, index=False)
        print(f"Processed data saved to: {output_path}")


def load_csv_data(filepath: str, 
                  has_header: bool = True,
                  zero_indexed: bool = True) -> Tuple[np.ndarray, List[int], List[str]]:
    """
    Convenience function to load data in one step.
    
    Parameters
    ----------
    filepath : str
        Path to CSV file
    has_header : bool, default=True
        Whether CSV has header row
    zero_indexed : bool, default=True
        Whether categories start from 0
        
    Returns
    -------
    X : np.ndarray
        Data matrix
    categories : list of int
        Number of categories per variable
    variable_names : list of str
        Variable names
    """
    loader = DataLoader(filepath)
    return loader.load_data(has_header=has_header, zero_indexed=zero_indexed)