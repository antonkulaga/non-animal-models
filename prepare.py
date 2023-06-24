#!/usr/bin/env python3

from pathlib import Path
from typing import Optional, List, OrderedDict

import click
from click import Context
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.schema import Document
from pycomfort.files import traverse

from nam.config import Locations
from nam.datasets import DatasetNAM


@click.group(invoke_without_command=False)
@click.pass_context
def app(ctx: Context):
    #if ctx.invoked_subcommand is None:
    #    click.echo('Running the default command...')
    #    test_index()
    pass

@app.command("download")
@click.option('--dataset', type=click.STRING, help="dataset folder from which read dataset")
@click.option('--threads', type=click.INT, default = 5, help="number of threads for the download")
@click.option('--base', type=click.STRING, default=".", help="base folder")
def download_command(dataset: str, threads: int, base: str):
    locations = Locations(Path(base))
    dataset_path: Path = locations.datasets / dataset
    assert dataset_path.exists(), f"dataset {dataset} does not exist at {str(dataset_path.absolute().resolve())}"
    assert (dataset_path / "models.tsv").exists(), f"dataset {dataset} does not have models.tsv at {str(dataset_path.absolute().resolve())}"
    current_dataset: DatasetNAM = DatasetNAM(dataset_path)
    print(f"downloading papers for {dataset_path}")
    return current_dataset.download(threads)

@app.command("parse")
@click.option('--dataset', type=click.STRING, help="dataset folder from which read dataset")
@click.option('--cores', type=click.INT, default = None, help="number of threads for the download")
@click.option('--strategy', type=click.STRING, default="fast", help="strategy")
@click.option('--base', type=click.STRING, default=".", help="base folder")
@click.option('--recreate_parent', type=click.BOOL, default=False, help="if parent folder should be recreated in the new destination")
def parse_command(dataset: str, cores: Optional[int], strategy: str, base: str, recreate_parent: bool):
    locations = Locations(Path(base))
    dataset_path: Path = locations.datasets / dataset
    where = dataset_path / "parsed_papers"
    where.mkdir(exist_ok=True, parents=True)
    assert dataset_path.exists(), f"dataset {dataset} does not exist at {str(dataset_path.absolute().resolve())}"
    assert (dataset_path / "models.tsv").exists(), f"dataset {dataset} does not have models.tsv at {str(dataset_path.absolute().resolve())}"
    current_dataset: DatasetNAM = DatasetNAM(dataset_path)
    print(f"downloading papers for {dataset_path}")
    return current_dataset.parse(where, strategy=strategy, cores=cores, recreate_parent = recreate_parent)

if __name__ == '__main__':
    app()