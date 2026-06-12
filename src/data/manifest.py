"""A manifest is the single source of truth for the dataset.

Replaces the *deprecated* HF dataset loading script (`ucf_crime.py`, which used a
`GeneratorBasedBuilder` + `trust_remote_code`). A manifest is just a list of
records — `{video_id, path, event, anomaly, split, size}` — built by walking a
directory (folder/filename encodes the class) or read from / written to a plain
`jsonl`/`parquet` file. No remote code, video-data friendly (paths only), and it
carries the ``event`` class label for free (VadCLIP / GS-MoE).

This mirrors what VadCLIP itself does: a CSV of ``{path, label}`` per video.
"""

import json
import os
from dataclasses import asdict, dataclass
from typing import Iterator, List, Optional

from src.data.labels import anomaly_label, event_id, parse_event


@dataclass
class Record:
    video_id: str
    path: str
    event: str  # one of UCF_CLASSES
    anomaly: str  # "Normal" | "Abnormal"
    split: str  # "train" | "test"
    size: int = 0  # bytes (or KiB), informational

    @property
    def class_id(self) -> int:
        from src.data.labels import CLASS_TO_ID

        return CLASS_TO_ID[self.event]

    @property
    def is_anomaly(self) -> bool:
        return self.anomaly == "Abnormal"


class Manifest:
    """An ordered collection of :class:`Record`."""

    def __init__(self, records: Optional[List[Record]] = None):
        self.records: List[Record] = list(records or [])

    # ---- builders ----
    @classmethod
    def from_filenames(cls, names: List[str], split: str) -> "Manifest":
        """Build from a list of file names (class parsed from each name)."""
        recs = []
        for name in names:
            vid = os.path.basename(name)
            recs.append(
                Record(
                    video_id=vid,
                    path=name,
                    event=parse_event(name),
                    anomaly=anomaly_label(name),
                    split=split,
                )
            )
        return cls(recs)

    @classmethod
    def from_dir(
        cls, root: str, split: str, exts: tuple = (".npy", ".mp4", ".avi")
    ) -> "Manifest":
        """Walk ``root`` recursively; the file name encodes the class."""
        recs = []
        for dirpath, _, files in os.walk(root):
            for f in sorted(files):
                if not f.endswith(exts):
                    continue
                p = os.path.join(dirpath, f)
                recs.append(
                    Record(
                        video_id=f,
                        path=p,
                        event=parse_event(f),
                        anomaly=anomaly_label(f),
                        split=split,
                        size=os.path.getsize(p),
                    )
                )
        return cls(recs)

    # ---- io ----
    def to_jsonl(self, path: str) -> None:
        with open(path, "w") as fp:
            for r in self.records:
                fp.write(json.dumps(asdict(r)) + "\n")

    @classmethod
    def from_jsonl(cls, path: str) -> "Manifest":
        recs = []
        with open(path) as fp:
            for line in fp:
                line = line.strip()
                if line:
                    recs.append(Record(**json.loads(line)))
        return cls(recs)

    def to_parquet(self, path: str) -> None:
        """Optional columnar format (needs pandas; recommended for the Hub)."""
        import pandas as pd

        pd.DataFrame([asdict(r) for r in self.records]).to_parquet(path, index=False)

    @classmethod
    def from_parquet(cls, path: str) -> "Manifest":
        import pandas as pd

        df = pd.read_parquet(path)
        return cls([Record(**row) for row in df.to_dict("records")])

    # ---- queries ----
    def filter(
        self, split: Optional[str] = None, anomaly: Optional[bool] = None
    ) -> "Manifest":
        recs = self.records
        if split is not None:
            recs = [r for r in recs if r.split == split]
        if anomaly is not None:
            recs = [r for r in recs if r.is_anomaly == anomaly]
        return Manifest(recs)

    def class_ids(self) -> List[int]:
        return [event_id(r.video_id) for r in self.records]

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[Record]:
        return iter(self.records)

    def __getitem__(self, i: int) -> Record:
        return self.records[i]
