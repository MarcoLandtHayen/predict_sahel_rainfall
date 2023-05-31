import numpy as np

# Define function to specify bar color, according to sign of values:
def bar_color(data,color_pos,color_neg):
    """Create array of colors to be used e.g., in plots.

    Parameters
    ----------
    data: Pandas DataFrame
        Containing numerical values.
    color_pos / color_neg: str
        Color codes to be assigned to positive and negative data values, respectively.
        
    Returns
    -------
    np.array
        Array of strings containing color codes.

    """
        
    return np.where(data.values>0,color_pos,color_neg).T