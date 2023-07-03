#!/usr/bin/env python3
import sys
from pathlib import Path
from typing import Optional, List, Dict
from typing import List, Dict, Any
import click
import json
from collections import defaultdict
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
@click.option('--count_only', type=click.BOOL, help= "If we write only counts to the json, and not metadata")
@click.option('--base', type=click.STRING, default=".", help="base folder")
@click.option('--log_level', type=click.Choice(LOG_LEVELS, case_sensitive=False), default=LogLevel.DEBUG.value, help="logging level")
def stats_command(dataset: str, access: str, count_only: bool, base: str,  log_level: str):
    configure_logger(log_level)
    locations = Locations(Path(base))
    dataset_path: Path = locations.datasets / dataset
    assert dataset_path.exists(), f"dataset {dataset} does not exist at {str(dataset_path.absolute().resolve())}"
    assert (dataset_path / "models.tsv").exists(), f"dataset {dataset} does not have models.tsv at {str(dataset_path.absolute().resolve())}"
    where = Path(access)
    logger.info(f"writing stats to {access}")
    current_dataset: DatasetNAM = DatasetNAM(dataset_path)
    current_dataset.open_access_stats(write_to=where, count_only = count_only)
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


def arxiv_dumps(where: Path):
    from paperscraper.get_dumps import biorxiv, medrxiv, chemrxiv
    medrxiv(save_path=str(where / "medrxiv"))  #  Takes ~30min and should result in ~35 MB file
    biorxiv(save_path=str(where / "biorxiv"))  # Takes ~1h and should result in ~350 MB file
    chemrxiv(save_path=str(where / "chemrxiv"))  #  Takes ~45min and should result in ~20 MB file


@app.command("arxiv_dumps")
@click.option('--folder', type=click.Path(), help="folder to save files")
@click.option('--log_level', type=click.Choice(["NONE", "DEBUG", "INFO", "ERROR", "WARNING", "DEBUG", "TRACE"], case_sensitive=False), default="debug", help="logging level")
def arxiv_dumps_command(folder: str, log_level: str):
    configure_logger(log_level)
    logger.info(f"downloading arxiv dumps to {folder}")
    where = Path(folder)
    where.mkdir(exist_ok=True, parents=True)
    return arxiv_dumps(where)


def sum_keys(json_files: List[str], output_path: Optional[str] = None) -> Dict[str, int]:
    """This function reads multiple JSON files and sums the keys with the same name and numeric types."""
    sum_dict: Dict[str, int] = defaultdict(int)

    for json_file in json_files:
        with open(json_file, 'r') as f:
            data: Dict[str, Any] = json.load(f)

        for key, value in data.items():
            # Ensure the value is numeric
            if isinstance(value, (int, float)):
                sum_dict[key] += value

    # If an output path is provided, write the sum_dict to a JSON file
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(sum_dict, f)

    return sum_dict


@app.command("stats_all")
@click.option('--jsons', '-j', multiple=True, type=click.Path(exists=True), help='Path to a JSON file.')
@click.option('--output', '-o', default=None, help='Path to output file.')
@click.option('--log_level', type=click.Choice(["NONE", "DEBUG", "INFO", "ERROR", "WARNING", "DEBUG", "TRACE"], case_sensitive=False), default="debug", help="logging level")
def stats_all_command(jsons: List[str], output: Optional[str], log_level: str) -> None:
    configure_logger(log_level)
    logger.info(f"LOAD STATS ALL WITH {jsons}")
    """This command line interface function invokes the sum_keys function and prints out the result."""
    sum_dict = sum_keys(jsons, output)
    logger.info(f"finished writing json to {output}")


if __name__ == '__main__':
    app()