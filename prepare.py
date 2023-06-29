#!/usr/bin/env python3
import sys
from pathlib import Path
from typing import Optional, List

from functional import seq
from getpaper.config import *
import click
from click import Context
from loguru import logger
from semanticscholar import SemanticScholar
from semanticscholar.Paper import Paper

from nam.config import Locations
from nam.datasets import DatasetNAM

@click.group(invoke_without_command=False)
@click.pass_context
def app(ctx: Context):
    #if ctx.invoked_subcommand is None:
    #    click.echo('Running the default command...')
    #    test_index()
    pass

@app.command("stats")
@click.option('--dataset', type=click.STRING, help="dataset folder from which read dataset")
@click.option('--access', type=click.Path(), help="help where to write statistics")
@click.option('--base', type=click.STRING, default=".", help="base folder")
@click.option('--log_level', type=click.Choice(LOG_LEVELS, case_sensitive=False), default=LogLevel.DEBUG.value, help="logging level")
def statistics_command(dataset: str, access: str,  base: str,  log_level: str):
    configure_logger(log_level)
    locations = Locations(Path(base))
    dataset_path: Path = locations.datasets / dataset
    assert dataset_path.exists(), f"dataset {dataset} does not exist at {str(dataset_path.absolute().resolve())}"
    assert (dataset_path / "models.tsv").exists(), f"dataset {dataset} does not have models.tsv at {str(dataset_path.absolute().resolve())}"
    where = Path(access)
    current_dataset: DatasetNAM = DatasetNAM(dataset_path)
    current_dataset.check_access(write_to=where)
    return where

@app.command("download")
@click.option('--dataset', type=click.STRING, help="dataset folder from which read dataset")
@click.option('--threads', type=click.INT, default = 5, help="number of threads for the download")
@click.option('--base', type=click.STRING, default=".", help="base folder")
@click.option('--log_level', type=click.Choice(LOG_LEVELS, case_sensitive=False), default=LogLevel.DEBUG.value, help="logging level")
def download_command(dataset: str, threads: int, base: str,  log_level: str):
    configure_logger(log_level)
    locations = Locations(Path(base))
    dataset_path: Path = locations.datasets / dataset
    assert dataset_path.exists(), f"dataset {dataset} does not exist at {str(dataset_path.absolute().resolve())}"
    assert (dataset_path / "models.tsv").exists(), f"dataset {dataset} does not have models.tsv at {str(dataset_path.absolute().resolve())}"
    current_dataset: DatasetNAM = DatasetNAM(dataset_path)
    logger.info(f"downloading papers for {dataset_path}")
    return current_dataset.download(threads)


@app.command("parse")
@click.option('--dataset', type=click.STRING, help="dataset folder from which read dataset")
@click.option('--cores', type=click.INT, default = None, help="number of threads for the download")
@click.option('--strategy', type=click.STRING, default="fast", help="strategy")
@click.option('--base', type=click.STRING, default=".", help="base folder")
@click.option('--recreate_parent', type=click.BOOL, default=False, help="if parent folder should be recreated in the new destination")
@click.option('--log_level', type=click.Choice(["NONE", "DEBUG", "INFO", "ERROR", "WARNING", "DEBUG", "TRACE"], case_sensitive=False), default="debug", help="logging level")
def parse_command(dataset: str, cores: Optional[int], strategy: str, base: str, recreate_parent: bool, log_level: str):
    configure_logger(log_level)
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