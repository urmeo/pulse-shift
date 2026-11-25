from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from . import config
from .provenance import atomic_output


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    with atomic_output(path) as output:
        frame.to_csv(output, index=False)


def write_table(df: pd.DataFrame | Iterable, name: str) -> pd.DataFrame:
    frame = df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)
    write_csv(frame, config.TABLES / f"{name}.csv")
    return frame
