from pathlib import Path
import polars as pl

from polars import Config
Config.set_fmt_str_lengths(5000)

class Locations:

    base: Path
    data: Path
    inputs: Path

    def __init__(self, base: Path):
        self.base = base
        self.data = self.base / "data"
        self.inputs = self.data / "inputs"
        self.datasets = self.inputs / "datasets"

