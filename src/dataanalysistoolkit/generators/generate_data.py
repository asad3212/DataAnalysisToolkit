"""generate_data.py

The data includes:

Random integers between 1 and 100 in columns 'A' and 'E'
Random choice of letters in column 'B'
Random floats from a normal distribution in column 'C'
Random choice of 1, 2, 3 in column 'D'
Random choice of fruits in column 'F'
Random floats from a uniform distribution in column 'G'
Random choice of 'x', 'y', 'z' in column 'H'
Some missing values have been introduced in columns 'C' and 'G'.
"""

import os
import numpy as np
import pandas as pd

def gen_data(n=100):
    """
    Generates a synthetic DataFrame with a mix of numeric, categorical, and
    missing data, useful for testing and demoing the toolkit.

    Args:
        n (int, optional): Number of data points to be created. Defaults to 100.

    Returns:
        pandas.DataFrame: The generated synthetic dataset.
    """
    # Generating random data
    data = {
        "A": np.random.randint(1, 100, n),  # Random integers between 1 and 100
        "B": np.random.choice(list("abcdefghij"), n),  # Random choice of letters
        "C": np.random.normal(0, 1, n),  # Random floats from a normal distribution
        "D": np.random.choice([1, 2, 3], n),  # Random choice of 1, 2, 3
        "E": np.random.randint(1, 100, n),  # Random integers between 1 and 100
        "F": np.random.choice(
            ["apple", "banana", "cherry", "date", "elderberry", "fig"], n
        ),  # Random choice of fruits
        "G": np.random.uniform(1, 5, n),  # Random floats from a uniform distribution
        "H": np.random.choice(["x", "y", "z"], n),  # Random choice of 'x', 'y', 'z'
    }

    # Converting to a DataFrame
    df = pd.DataFrame(data)

    # Introducing some missing values in columns 'C' and 'G'
    df.loc[np.random.choice(df.index, size=int(0.1 * n)), "C"] = np.nan
    df.loc[np.random.choice(df.index, size=int(0.1 * n)), "G"] = np.nan

    return df


def save_data(df, path="data/test_random.csv"):
    """
    Saves a generated DataFrame to a CSV file, creating the parent directory
    if it doesn't already exist.

    Args:
        df (pandas.DataFrame): The DataFrame to save.
        path (str, optional): Destination CSV path. Defaults to
        "data/test_random.csv".
    """
    parent_dir = os.path.dirname(path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    df.to_csv(path, index=False)


# Module-level DataFrame available for import (e.g. `from generate_data import df`),
# generated once when this module is first imported.
df = gen_data()


if __name__ == "__main__":
    save_data(df)
