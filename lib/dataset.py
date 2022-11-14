import re
import pandas as pd


def clean(x: str) -> str:
    """
    Clean the string from basic separators '-', '/', '_'
    Remove words which are completely numbers

    Args:
        x: text to clean

    Returns:

    """
    x = x.replace('-', ' ').replace('_', ' ').replace('/', ' ')
    return re.sub(r'\b[0-9]+\b\s*', '', x).strip()  # remove words that are just numbers


# truncate string
def clean_and_truncate(x: str, max_len: int, pad: str = '[UNK]') -> str:
    """
    clean and truncate the input description

    Args:
         x: text to process
         max_len: maximum word length for truncation
         pad: character to use for padding if empty
    """
    if x and not pd.isnull(x):
        trunc = ' '.join(clean(x).split(' ')[:max_len])
        return trunc.strip()
    return ' '.join([pad] * max_len).strip()
