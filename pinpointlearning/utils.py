"""
Utility functions for creating the exam analysis tools.
"""

import pandas as pd
import numpy as np


def load_sample_data(
    n_cols: int = 24, n_rows: int = 10**3, binary: bool = False, synthetic=True
):
    """Returns an array of sample data for use in prototyping the analysis
    pipelines

    Args:
        n_cols (int, optional): Number of columns to return. Columsn represent
        independent questions. Defaults to 24.
        n_rows (_type_, optional): Number of rows to return. Rows represent
         individual students. Defaults to 10**3.
        binary (bool, optional): Whether to return raw scores or binarised
        pass/fail data. Defaults to False.
        synthetic (bool, option): Whether to load the read data or generate
         random placeholder values. Defaults to False

    Returns:
        _type_: _description_
    """
    if synthetic:
        data = np.random.uniform(0, 7, (n_rows, n_cols))
    else:
        df = pd.read_csv(
            "../data/LargeDataSet13_HigherExamsWithClassAndSchoolsAvailable.csv",
            skiprows=7,
            usecols=range(1, n_cols + 1),
        )
        data = df.to_numpy()
        data = np.nan_to_num(data)
        data = data[np.amax(data, axis=1) <= 7, :]
        data = data[:n_rows]

    if binary:
        data = data / np.amax(data, axis=0)
        data = data > 0.5
    return data
