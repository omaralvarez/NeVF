from time import perf_counter
import json
import datetime as dt

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
import humanize
import numpy as np


def ic_rich(s):
    console.print(s)


EXPAND = False
TRANSIENT = True
IND_COLUMNS = [
    TextColumn("🔄 [progress.description]{task.description}"),
    BarColumn(),
]

PROG_COLUMNS = [
    TextColumn("🔄 [progress.description]{task.description}"),
    TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
    BarColumn(),
    MofNCompleteColumn(),
    TextColumn("•"),
    TimeElapsedColumn(),
    TextColumn("•"),
    TimeRemainingColumn(),
]


class ProgressBar:
    def __init__(
        self,
        indeterminate: bool = False,
        transient: bool = TRANSIENT,
        disable: bool = False,
    ) -> None:
        # Define custom progress bar
        if indeterminate:
            self.progress = Progress(
                *IND_COLUMNS,
                transient=transient,
                console=console,
                disable=disable,
                expand=EXPAND,
            )
        else:
            self.progress = Progress(
                *PROG_COLUMNS,
                transient=transient,
                console=console,
                disable=disable,
                expand=EXPAND,
            )


class Timer:
    def __init__(self) -> None:
        self._start = None
        self._stop = None
        self.elapsed_s = None
        self.history = []

    def start(self) -> None:
        self._start = perf_counter()

    def stop(self) -> None:
        self._stop = perf_counter()

    def elapsed(self) -> None:
        self.elapsed_s = self._stop - self._start
        self.history.append(self.elapsed_s)
        console.print(
            "⏳ Elapsed time: {}".format(
                humanize.precisedelta(
                    dt.timedelta(seconds=self.elapsed_s), format="%0.4f"
                )
            )
        )


class Config:
    def __init__(self, path) -> None:
        with open(path, "r") as jsonfile:
            self._config = json.load(jsonfile)

        console.print("✅ Config {} loaded...".format(path))

    def __getitem__(self, key):
        return self._config[key]

    def __setitem__(self, key, data):
        self._config[key] = data

    def exists(self, key: str) -> bool:
        return True if key in self._config else False

    def to_dict(self) -> dict:
        return self._config


# General
SEED = 42
TYPE = np.float64

# Rich
SPINNER = "aesthetic"
REFRESH = 20
EXPAND = False

console = Console()

rng = np.random.default_rng(seed=SEED)
