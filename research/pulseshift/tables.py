from collections.abc import Iterable

import pandas as pd

from . import config
from .provenance import atomic_output


def write_table(df: pd.DataFrame | Iterable, name: str) -> pd.DataFrame:
    frame = df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
    with atomic_output(config.TABLES / f"{name}.csv") as output:
        frame.to_csv(output, index=False)
    return frame
