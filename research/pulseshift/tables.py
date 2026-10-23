from collections.abc import Iterable

import pandas as pd

from . import config
from .provenance import atomic_output


def write_table(
    df: pd.DataFrame | Iterable, name: str, tables_dir=None
) -> pd.DataFrame:
    frame = df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
    tables_dir = tables_dir or config.TABLES
    with atomic_output(tables_dir / f"{name}.csv") as output:
        frame.to_csv(output, index=False)
    return frame
