import json
from collections import OrderedDict
from pathlib import Path
from typing import List, Optional
from getpaper.download import check_access
import polars as pl
import tiktoken
from functional import seq
from getpaper.download import download_papers
from getpaper.parse import parse_papers
from loguru import logger
from semanticscholar.Paper import Paper


def read_tsv(path: Path):
    return pl.read_csv(path, sep="\t", infer_schema_length=10000, null_values=["N/A", "-", "null", "None"])


class DatasetNAM:

    models: pl.DataFrame
    fields: pl.DataFrame
    dropdowns: pl.DataFrame
    papers_folder: Path
    parsed_papers_folder: Path
    index_folder: Path
    dois: List[str]
    default_model: str

    @property
    def doi_cols(self):
        result = [c for c in self.models.columns if "DOI" in c]
        assert len(result) == 1, f"there should be only one DOI column, but there are {result}"
        return result

    def __init__(self, folder: Path, default_model: str = "gpt-3.5-turbo-16k"):
        self.default_model = default_model
        self.encoding = tiktoken.encoding_for_model(self.default_model)
        self.models = read_tsv(folder / "models.tsv")
        self.dois = [d.replace("https://doi.org/", "") for d in self.models[self.doi_cols[0]].unique().to_list()]
        self.dropdowns = read_tsv(folder / "dropdowns.tsv")
        self.fields = read_tsv(folder / "fields.tsv")
        self.papers_folder = folder / "papers"
        self.papers_folder.mkdir(exist_ok=True, parents=True)
        self.parsed_papers_folder = folder / "parsed_papers"
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
            return self.parsed_papers_folder / (doi+".txt")
        def exists(string: str):
            return Path(string).exists()

        def load_text(url: str):
            where = Path(url)
            if not where.exists():
                return None
            return where.read_text("utf-8")

        def num_tokens(text: Optional[str]):
            if text is None:
                return None
            return len(self.encoding.encode(text))

        doi_col = pl.col(self.doi_cols[0])
        path_col = doi_col.apply(doi_to_path).alias("path")
        exist_col = path_col.apply(exists).alias("exists")
        parsed_path_col = doi_col.apply(doi_to_parsed_path).alias("parsed_path")
        parsed_exist_col = parsed_path_col.apply(exists).alias("parsed_exists")
        text_col = parsed_path_col.apply(load_text).alias("text")
        tokens_number = text_col.apply(num_tokens).alias("token_number")
        return self.models.with_columns([path_col, exist_col, parsed_path_col, parsed_exist_col, tokens_number, text_col])

    def validate_downloads(self):
        total = len(self.dois)
        i = 0
        for doi in self.dois:
            if (self.papers_folder / doi).exists():
                i = i + 1
        logger.info(f"DOWNLOADED DOIS {i} out of {total}")

    def check_access_paginated(self, page: int = 500):
        if len(self.dois) < page:
            return check_access(self.dois)
        else:
            slides = seq(self.dois).sliding(page, page).to_list()
            pairs = [check_access(s.to_list()) for s in slides]
            opened = seq([p[0] for p in pairs]).flatten().to_list()
            closed = seq([p[1] for p in pairs]).flatten().to_list()
            return opened, closed

    def open_access_stats(self, write_to: Optional[Path] = None, count_only: bool = True):
        opened, closed = self.check_access_paginated()
        total = len(opened) + len(closed)
        result = {
            "total_papers": total,
            "total_open": len(opened),
            "total_closed": len(closed),
            "not_found": len(self.dois) - total
        }
        if not count_only:
            result["opened"] = [p for p in opened]
            result["closed"] = [p for p in closed]
        if write_to is not None:
            write_to.touch(exist_ok=True)
            json_data = json.dumps(result)
            write_to.write_text(json_data)
        return result

    def download(self, threads: int = 5):
        """
        Downloads papers from the models
        :return:
        """
        results: OrderedDict = download_papers(self.dois, self.papers_folder, threads=threads)
        succeeded: OrderedDict[str, (str, Path, Path)] = results[0]
        failed = results[1]
        for f in failed:
            logger.warning(f"download failed for {f}")
        good = len(succeeded)
        bad = len(failed)
        logger.info(f"TOTAL DOWNLOADED [{good}/{good+bad}], FAILED: {bad}")
        return succeeded

    def parse(self, destination: Optional[Path], strategy: str = "auto", cores: Optional[int] = None,  recreate_parent = True):
        return parse_papers(self.papers_folder, destination, strategy=strategy, cores=cores, recreate_parent=recreate_parent)

    def index(self):
        return self.papers_folder

    def get_dropdowns_dictionary(self, columns: Optional[List[str]] = None) -> OrderedDict[str, List[str]]:
        cols = self.dropdowns.columns if columns is None else columns
        lists = [(col.lower().replace(" ", "_").replace("_/_", "_or_"), self.dropdowns.filter(pl.col(col).is_not_null()).select(col).to_series().to_list()) for col in cols]
        return OrderedDict(lists)