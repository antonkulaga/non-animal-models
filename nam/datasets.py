from pathlib import Path

import getpaper.parse
import polars as pl
from typing import List, OrderedDict, Optional
from getpaper.download import download_papers
from getpaper.parse import parse_papers

def read_tsv(path: Path):
    return pl.read_csv(path, sep="\t", infer_schema_length=10000, null_values=["N/A", "-", "null", "None"])

class DatasetNAM:


    models: pl.DataFrame
    fields: pl.DataFrame
    dropdowns: pl.DataFrame
    papers_folder: Path
    parsed_folder: Path
    index_folder: Path
    dois: List[str]

    @property
    def doi_cols(self):
        result = [c for c in self.models.columns if "DOI" in c]
        assert len(result) == 1, f"there should be only one DOI column, but there are {doi_cols}"
        return result

    def __init__(self, folder: Path):
        self.models = read_tsv(folder / "models.tsv")
        self.dois = self.models[self.doi_cols[0]].unique().to_list()
        self.dropdowns = read_tsv(folder / "dropdowns.tsv")
        self.fields = read_tsv(folder / "fields.tsv")
        self.papers_folder = folder / "papers"
        self.parsed_folder = folder / "parsed_papers"
        self.papers_folder.mkdir(exist_ok=True, parents=True)
        self.index_folder = folder / "index"
        self.index_folder.mkdir(exist_ok=True, parents=True)



    @property
    def extended_models(self):
        """
        Model extended with downloaded data
        :return:
        """
        def doi_to_path(doi: str):
            return self.papers_folder / (doi+".pdf")

        def doi_to_parsed_path(doi: str):
            return self.papers_folder / (doi+".txt")
        def exists(string: str):
            return Path(string).exists()
        doi_col = pl.col(self.doi_cols[0])
        path_col = doi_col.apply(doi_to_path).alias("path")
        exist_col = path_col.apply(exists).alias("exists")
        return self.models.with_columns([path_col, exist_col])

    def validate_downloads(self):
        total = len(self.dois)
        i = 0
        for doi in self.dois:
            if (self.papers_folder / doi).exists():
                i = i + 1
        print(f"DOWNLOADED DOIS {i} out of {total}")

    def download(self, threads: int = 5):
        """
        Downloads papers from the models
        :return:
        """
        results = download_papers(self.dois, self.papers_folder, threads=threads)
        succeeded: OrderedDict[str, Path] = results[0]
        failed = results[1]
        for f in failed:
            print(f"download failed for {f}")
        good = len(succeeded)
        bad = len(failed)
        print(f"TOTAL DOWNLOADED [{good}/{good+bad}], FAILED: {bad}")
        return succeeded

    def parse(self, destination: Optional[Path], strategy: str = "auto", cores: Optional[int] = None,  recreate_parent = True):
        return getpaper.parse.parse_papers(self.papers_folder, destination, strategy=strategy, cores=cores, recreate_parent=recreate_parent)

    def index(self):
        return self.papers_folder
