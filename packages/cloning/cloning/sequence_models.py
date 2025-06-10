import io
import os
import uuid
from enum import StrEnum, auto
from itertools import groupby
from typing import ClassVar

from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqFeature import CompoundLocation, FeatureLocation, SeqFeature
from Bio.SeqRecord import SeqRecord
from pydantic import BaseModel, Field, JsonValue, field_validator, model_validator

from .enzymes import ENZYME_MATCHER

# annotate takes the longest - usually > 10
DEFAULT_TIMEOUT = 20
ALOT_OF_DNA = 25
ALOT_OF_ANNOTATIONS = 5
try:
    API_URL = os.environ["AVIARY_CLONING_MODAL_URL"]
except KeyError:
    raise ValueError(
        "AVIARY_CLONING_MODAL_URL is not set. Please set it to the URL of the Modal app."
    ) from None


class SearchResult(BaseModel):
    title: str
    body: str
    docid: str
    featureid: int
    sequence: str
    score: float = 0.0
    genbank: str

    @field_validator("title")
    @classmethod
    def title_must_not_be_empty(cls, v: str) -> str:
        if not v:
            return make_pretty_id("seq")
        return v

    def to_sequence(self) -> "BioSequence":
        return BioSequence.from_genbank(self.genbank)


def make_pretty_id(prefix) -> str:
    start = str(uuid.uuid4())
    frags = start.split("-")
    return prefix + "-" + "".join(frags[:2])


class SequenceType(StrEnum):
    DNA = auto()
    RNA = auto()
    PROTEIN = auto()


def _guess_sequence_type(sequence: str) -> SequenceType:
    if set(sequence) <= set("ACGT"):
        return SequenceType.DNA
    if set(sequence) <= set("ACGU"):
        return SequenceType.RNA
    if "O" in sequence or "U" in sequence:
        raise ValueError(f"Invalid character in sequence {sequence}")
    return SequenceType.PROTEIN


class Annotation(BaseModel):
    start: int
    end: int
    type: str
    strand: int
    name: str
    full_name: str | None = None


class BioSequence(BaseModel):
    sequence: str
    type: SequenceType
    is_circular: bool = False
    name: str = Field(default_factory=lambda: make_pretty_id("seq"))
    description: str | None = None
    annotations: list[Annotation] | None = None

    RESTRICTION_SITE: ClassVar[str] = "restriction site"

    def make_state_dict(self) -> dict:
        # Abbreviated sequence
        seq_len = len(self.sequence)
        seq_str = (
            self.sequence
            if seq_len <= 2 * ALOT_OF_DNA
            else f"{self.sequence[:ALOT_OF_DNA]}...{self.sequence[-ALOT_OF_DNA:]}"
        )

        state_dict: dict[str, JsonValue] = {
            "type": f"BioSequence({self.type.value}, {'Circular' if self.is_circular else 'Linear'})",
            "name": self.name,
            "sequence": f"{seq_len} bp: {seq_str}",
        }

        if self.description:
            state_dict["description"] = self.description

        if self.annotations:
            state_dict["annotations"] = [
                f"{ann.type} | {ann.name}: {ann.start}-{ann.end}, {ann.full_name}"
                for ann in self.get_ordered_annotations()
                if ann.type != BioSequence.RESTRICTION_SITE
            ]

        return state_dict

    def __str__(self):
        return self._make_str()

    def _make_str(self, list_restrictions: bool = False) -> str:
        # Header with name and description
        header = f"{self.name}:"
        if self.description:
            header += f" {self.description}"

        # Sequence type and circularity
        seq_info = (
            f"Type: {self.type.value}, {'Circular' if self.is_circular else 'Linear'}"
        )

        # Abbreviated annotations
        ann_str = ""
        if self.annotations:
            ann_str = "Annotations:\n"
            for ann in (
                self.annotations
                if list_restrictions
                else list(
                    filter(
                        lambda x: x.type != BioSequence.RESTRICTION_SITE,
                        self.annotations,
                    )
                )
            ):
                ann_str += f"  {ann.type} | {ann.name}: {ann.start}-{ann.end}, {ann.full_name}\n"

        # Abbreviated sequence
        seq_len = len(self.sequence)
        seq_str = (
            self.sequence
            if seq_len <= 2 * ALOT_OF_DNA
            else f"{self.sequence[:ALOT_OF_DNA]}...{self.sequence[-ALOT_OF_DNA:]}"
        )

        # Combine all parts
        return f"{header}\n{seq_info}\n{ann_str}Sequence ({seq_len} bp): {seq_str}"

    def get_ordered_annotations(self) -> list[Annotation]:
        if not self.annotations:
            return []
        return sorted(self.annotations, key=lambda x: x.start)

    @field_validator("sequence")
    @classmethod
    def sequence_should_be_plausible(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError(f"Sequence cannot be empty. Got: {v}")
        if any(not c.isalpha() for c in v):
            raise ValueError(f"Sequence must only contain letters. Got: {v}")
        return v

    @classmethod
    def from_fasta(cls, fasta_record: str) -> "BioSequence":
        header, *seqs = fasta_record.strip().split("\n")
        sequence = "".join(seqs)
        header = header[1:]
        if ":" in header:
            name_parts = header.split(":", 1)
            name = name_parts[0].strip()
            description = name_parts[1].strip() if len(name_parts) > 1 else None
        elif "|" in header:
            name_parts = header.split("|", 1)
            name = name_parts[0].strip()
            description = name_parts[1].strip() if len(name_parts) > 1 else None
        else:
            name = header.split(" ")[0].strip()
            description = " ".join(header.split(" ")[1:]).strip() or None
        is_circular = False
        if description and "(circular)" in description:
            is_circular = True
            description = description.replace("(circular)", "").strip()
        return cls(
            sequence=sequence.replace("\n", "").strip(),
            type=_guess_sequence_type(sequence),
            is_circular=is_circular,
            name=name,
            description=description,
            annotations=[],
        )

    def to_fasta(self) -> str:
        header = f">{self.name or 'Sequence'}"
        if self.description:
            header += f": {self.description}"
        if self.is_circular:
            header += " (circular)"
        # make sure header is on one line
        header = header.replace("\n", " ")
        return f"{header}\n{self.sequence}\n"

    @classmethod
    def from_genbank(cls, gb_record: str) -> "BioSequence":
        # write the record to a temporary file and read it back
        with io.StringIO(gb_record) as f:
            record = SeqIO.read(f, "genbank")
        sequence = str(record.seq)

        # look high and low for indications of circularity
        is_circular = record.annotations.get(
            "topology", ""
        ).lower() == "circular" or any(
            qualifier.lower() == "circular"
            for feature in record.features
            if feature.type == "source"
            for qualifier in feature.qualifiers.get("topology", [])
        )
        annotations = []

        for feature in record.features:
            if isinstance(feature.location, CompoundLocation):
                start = int(feature.location.parts[0].start)
                end = int(feature.location.parts[-1].end)
            else:
                start = int(feature.location.start)
                end = int(feature.location.end)

            annotations.append(
                Annotation(
                    start=start,
                    end=end,
                    strand=feature.location.strand,
                    type=feature.type,
                    name=feature.qualifiers.get("note", [""])[0].replace("|", "/"),
                    full_name=feature.qualifiers.get("full_name", [""])[0].split(" (")[
                        0
                    ],
                )
            )
        return cls(
            sequence=sequence,
            type=_guess_sequence_type(sequence),
            is_circular=is_circular,
            name=record.name,
            description=record.description,
            annotations=annotations,
        )

    def to_genbank(self) -> str:
        record = SeqRecord(
            Seq(self.sequence),
            id=self.name or "Unknown",
            name=self.name or "Unknown",
            description=self.description or "",
            annotations={
                "topology": "circular" if self.is_circular else "linear",
                "molecule_type": "DNA",
            },
        )

        seq_length = len(self.sequence)
        for ann in self.annotations or []:
            feature = _create_feature(ann, seq_length)
            record.features.append(feature)

        with io.StringIO() as f:
            SeqIO.write(record, f, "genbank")
            f.seek(0)
            return f.read()

    def _annotate_enzyme_sites(self) -> None:
        if not self.annotations:
            self.annotations = []
        counts: dict[str, int] = {}
        for end_idx, (enzyme, is_forward) in ENZYME_MATCHER.iter(self.sequence):
            a = Annotation(
                start=end_idx
                - (
                    len(enzyme["regexp_for"])
                    if is_forward
                    else len(enzyme["regexp_rev"])
                ),
                end=end_idx,
                strand=1 if is_forward else -1,
                name=enzyme["name"],
                type=BioSequence.RESTRICTION_SITE,
                full_name=f"Digestion site for {enzyme['name']} - recognition site: {enzyme['recognition_site']}",
            )
            self.annotations.append(a)
            counts[enzyme["name"]] = counts.get(enzyme["name"], 0) + 1

        # now we filter out all annotations that resulted in a ridiculous number of hits
        self.annotations = [
            ann
            for ann in self.annotations
            if counts.get(ann.name, 0) < ALOT_OF_ANNOTATIONS
        ]

    def annotate_restriction_sites(self) -> str:
        if not self.annotations or not any(
            ann.type == BioSequence.RESTRICTION_SITE for ann in self.annotations
        ):
            self._annotate_enzyme_sites()
        return self._make_str(list_restrictions=True)

    def view_statistics(self) -> str:
        """View GC content, max homopolymer repeat length, seq length."""
        gc_content = round(
            (self.sequence.count("G") + self.sequence.count("C"))
            / len(self.sequence)
            * 100
        )
        max_repeat = max(len(list(g)) for _, g in groupby(self.sequence))
        stats = f"Length: {len(self.sequence)}\n"
        if self.type == SequenceType.DNA:
            stats += f"GC content: {gc_content:d}%\n"
        stats += f"Longest repeat: {max_repeat}"
        return stats


def _create_feature(ann: Annotation, seq_length: int) -> SeqFeature:
    location = (
        FeatureLocation(ann.start, ann.end, strand=ann.strand)
        if ann.end <= seq_length
        else CompoundLocation([
            FeatureLocation(ann.start, seq_length, strand=ann.strand),
            FeatureLocation(0, ann.end - seq_length, strand=ann.strand),
        ])
    )

    qualifiers = {
        "note": ann.name.replace("/", "|"),
        "full_name": str(ann.full_name).replace("/", "|"),
    }

    return SeqFeature(location, type=ann.type, qualifiers=qualifiers)


class BioSequences(BaseModel):
    sequences: list[BioSequence] = Field(default_factory=list)

    def make_state_dict(self) -> dict:
        return {
            "type": "BioSequences",
            "sequences": [seq.make_state_dict() for seq in self.sequences],
        }

    def __str__(self):
        return "\n" + "\n\n".join(str(seq) for seq in self.sequences)

    @model_validator(mode="after")
    def check_sequence_types(self) -> "BioSequences":
        if not self.sequences:
            return self
        first_type = self.sequences[0].type
        if any(seq.type != first_type for seq in self.sequences):
            raise ValueError("All BioSequences must be of the same type")
        return self

    def to_fasta(self) -> str:
        return "".join(seq.to_fasta() for seq in self.sequences)

    @classmethod
    def from_fasta(cls, fasta_content: str) -> "BioSequences":
        fasta_records = fasta_content.strip().split("\n>")
        sequences = [
            BioSequence.from_fasta(">" + record if i > 0 else record)
            for i, record in enumerate(fasta_records)
            if record.strip()
        ]
        return cls(sequences=sequences)


class OptimizationResult(BaseModel):
    input_sequence: str
    input_type: str
    optimized_dna: str
    changes: list[dict]
    initial_gc_content: float
    optimized_gc_content: float
    codon_table_index: int


class ORF(BaseModel):
    start: int
    end: int
    strand: str
    frame: int
    sequence: str
    translation: str


class FragmentOutput(BaseModel):
    sequence: str
    forward_overhang: str
    reverse_overhang: str


class EnzymeOutput(BaseModel):
    name: str
    regexp_for: str
    regexp_rev: str
    skip: int
    overhead_length: int
    recognition_site: str


class PCRResult(BaseModel):
    forward_primer: str
    reverse_primer: str
    amplicon_fasta: str
