from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from Bio import SeqIO
from Bio.SeqFeature import CompoundLocation, FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord

# This is the number of columns in the file
# which  comes from the --format-output option
# in the convertalis mmseqs command that
# creates the result files
FRAME_COL_NUM: int = 8
NO_FRAME_COL_NUM: int = 7


@dataclass
class Annotation:
    name: str
    e_score: float
    start: int
    end: int
    frame: int | None
    strand: Literal[1, -1]
    qcov: float
    tcov: float
    full_name: str

    @classmethod
    def read_annotations(cls, annotation_file: str | Path) -> list["Annotation"]:
        annotations: list = []
        with open(annotation_file, encoding="utf-8") as file:
            for line_number, line in enumerate(file, 1):
                fields = line.strip().split("\t")
                if not (
                    len(fields) == FRAME_COL_NUM or len(fields) == NO_FRAME_COL_NUM
                ):
                    raise ValueError(
                        f"Error on line {line_number}: Expected {FRAME_COL_NUM} tab-separated fields, but got {len(fields)}.\n"
                        f"Line content: {line.strip()}"
                    )
                try:
                    if len(fields) == FRAME_COL_NUM:
                        name = fields[0].split(",")[0]
                        e_score = float(fields[1])
                        pos1 = int(fields[2])
                        pos2 = int(fields[3])
                        frame = int(fields[4])
                        qcov = float(fields[5])
                        tcov = float(fields[6])
                        full_name = fields[7]
                    else:
                        name = fields[0].split(",")[0]
                        e_score = float(fields[1])
                        pos1 = int(fields[2])
                        pos2 = int(fields[3])
                        frame = None
                        qcov = float(fields[4])
                        tcov = float(fields[5])
                        full_name = fields[6]

                    start = min(pos1, pos2) - 1
                    end = max(pos1, pos2)
                    strand: Literal[1, -1] = 1 if pos1 < pos2 else -1
                except ValueError as e:
                    raise ValueError(
                        f"Error on line {line_number}: {e!s}.\n"
                        f"Line content: {line.strip()}"
                    ) from e

                if e_score < 0:
                    raise ValueError(
                        f"Error on line {line_number}: E-score ({e_score}) cannot be negative."
                    )

                annotations.append(
                    cls(name, e_score, start, end, frame, strand, qcov, tcov, full_name)
                )

        return annotations


def compute_overlap(
    ann1: Annotation, ann2: Annotation, min_overlap: float = 0.8
) -> bool:
    overlap_start = max(ann1.start, ann2.start)
    overlap_end = min(ann1.end, ann2.end)
    overlap_length = max(0, overlap_end - overlap_start)

    overlap_percent1 = overlap_length / (ann1.end - ann1.start)
    overlap_percent2 = overlap_length / (ann2.end - ann2.start)

    return max(overlap_percent1, overlap_percent2) > min_overlap


def filter_annotations(
    annotations: list[Annotation], e_score_threshold: float, tcov_threshold: float
) -> list[Annotation]:
    annotations.sort(key=lambda x: (-x.tcov, -(x.end - x.start)))

    filtered: list[Annotation] = []
    for ann in annotations:
        if ann.e_score > e_score_threshold or ann.e_score == 0:
            continue
        if ann.tcov < tcov_threshold:
            continue

        overlap: bool = any(
            compute_overlap(ann, filtered_ann) for filtered_ann in filtered
        )

        if not overlap:
            filtered.append(ann)

    return filtered


def adjust_circular_annotation(ann: Annotation, seq_length: int) -> Annotation | None:
    """This will adjust the indices for the circular case, so that they do not exceed seq_length. They will wrap."""
    if ann.start >= seq_length:
        return None
    if ann.end <= seq_length:
        return ann
    a = Annotation(**ann.__dict__)
    a.end = 0 + (ann.end - seq_length)
    return a


def create_feature(ann: Annotation, seq_length: int) -> SeqFeature:
    location = (
        FeatureLocation(ann.start, ann.end, strand=ann.strand)
        if ann.start <= ann.end  # wrapped annotations
        else CompoundLocation([
            FeatureLocation(ann.start, seq_length, strand=ann.strand),  # up to end
            FeatureLocation(0, ann.end, strand=ann.strand),  # wrapped around
        ])
    )

    qualifiers = {
        "note": ann.name.replace("/", "|"),
        "full_name": f"{ann.full_name} ({ann.e_score:.2e}, tcov = {ann.tcov:.2f})"
        + (f"frame = {ann.frame}" if ann.frame else "").replace("/", "|"),
    }

    return SeqFeature(location, type=ann.name, qualifiers=qualifiers)


def create_genbank_record(
    fasta_file: str | Path, annotations: list[Annotation], circular: bool
) -> SeqRecord:
    records: list[SeqRecord] = list(SeqIO.parse(fasta_file, "fasta"))
    if not records:
        raise ValueError("No sequences found in the FASTA file")
    record: SeqRecord = records[0]

    # Note: we had to double it at an earlier stage, when we actually get the annotations
    seq_length: int = len(record.seq) // 2 if circular else len(record.seq)
    record.seq = record.seq[:seq_length]

    adjusted_annotations: list[Annotation] = []
    if circular:
        adjusted_annotations = [
            a
            for ann in annotations
            if (a := adjust_circular_annotation(ann, seq_length))
        ]
    else:
        adjusted_annotations = annotations

    for ann in adjusted_annotations:
        if ann.start < 0 or ann.end > seq_length:
            print(
                f"Warning: Annotation {ann.name} ({ann.start + 1}-{ann.end}) is out of sequence bounds (1-{seq_length}). Skipping.",
                file=__import__("sys").stderr,
            )
            continue
        feature: SeqFeature = create_feature(ann, seq_length)
        record.features.append(feature)
    if circular:
        record.annotations["topology"] = "circular"
    record.annotations["molecule_type"] = "DNA"
    return record


def annotation2genbank(
    annotations: str | Path,
    fasta: str | Path,
    output: str | Path,
    e_score_threshold: float,
    tcov_threshold: float,
    circular: bool,
) -> None:
    all_annotations: list[Annotation] = Annotation.read_annotations(annotations)
    filtered_annotations: list[Annotation] = filter_annotations(
        all_annotations, e_score_threshold, tcov_threshold
    )

    record: SeqRecord = create_genbank_record(fasta, filtered_annotations, circular)

    SeqIO.write(record, output, "genbank")
    print(f"GenBank file '{output}' has been created successfully.")
