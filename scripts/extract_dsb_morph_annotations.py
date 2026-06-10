#!/usr/bin/env python3
"""Extract Lower Sorbian morphology annotations from an Apertium metadix."""

from __future__ import annotations

import argparse
import math
import random
import sys
import xml.etree.ElementTree as ET
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Annotation:
    surface: str
    morphs: tuple[str, ...]
    paradigm: str


def element_text(elem: ET.Element | None) -> str:
    if elem is None:
        return ""
    return "".join(elem.itertext())


def generate_forms(
    stem: str,
    pardef_name: str,
    pardefs: dict[str, ET.Element],
) -> list[tuple[str, tuple[str, ...]]]:
    pardef = pardefs.get(pardef_name)
    if pardef is None:
        return []

    forms: list[tuple[str, tuple[str, ...]]] = []
    for entry in pardef.findall("e"):
        if entry.get("r") == "LR":
            continue

        left = element_text(entry.find("p/l"))
        inner_par = entry.find("par")
        if inner_par is not None:
            forms.extend(generate_forms(stem + left, inner_par.get("n", ""), pardefs))
            continue

        surface = stem + left
        if not surface:
            continue
        if stem and left:
            forms.append((surface, (stem, left)))
        elif left:
            forms.append((surface, (left,)))
        else:
            forms.append((surface, (stem,)))

    return forms


def extract_annotations(dix_path: Path) -> list[Annotation]:
    root = ET.parse(dix_path).getroot()
    pardefs = {
        pardef.get("n", ""): pardef
        for pardef in root.findall(".//pardefs/pardef")
    }

    by_surface: OrderedDict[str, Annotation] = OrderedDict()
    for entry in root.findall('.//section[@id="main"]/e'):
        if entry.get("r") == "LR":
            continue

        stem = element_text(entry.find("i"))
        par = entry.find("par")
        if par is None:
            continue

        paradigm = par.get("n", "")
        for surface, morphs in generate_forms(stem, paradigm, pardefs):
            if surface not in by_surface:
                by_surface[surface] = Annotation(surface, morphs, paradigm)

    return list(by_surface.values())


def balanced_sample(
    annotations: list[Annotation],
    sample_size: int,
    seed: int,
) -> list[Annotation]:
    if sample_size >= len(annotations):
        return annotations

    rng = random.Random(seed)
    groups: dict[str, list[Annotation]] = defaultdict(list)
    for annotation in annotations:
        groups[annotation.paradigm].append(annotation)

    shuffled_groups = []
    for paradigm, records in groups.items():
        records = records[:]
        rng.shuffle(records)
        shuffled_groups.append((paradigm, records))

    weights = {paradigm: math.sqrt(len(records)) for paradigm, records in shuffled_groups}
    total_weight = sum(weights.values())
    quotas: dict[str, int] = {}
    remainders: list[tuple[float, str]] = []

    for paradigm, records in shuffled_groups:
        raw_quota = sample_size * weights[paradigm] / total_weight
        quota = min(len(records), int(raw_quota))
        quotas[paradigm] = quota
        remainders.append((raw_quota - quota, paradigm))

    remaining = sample_size - sum(quotas.values())
    for _, paradigm in sorted(remainders, reverse=True):
        if remaining == 0:
            break
        capacity = len(groups[paradigm]) - quotas[paradigm]
        if capacity <= 0:
            continue
        quotas[paradigm] += 1
        remaining -= 1

    sample: list[Annotation] = []
    for paradigm, records in sorted(shuffled_groups):
        sample.extend(records[: quotas[paradigm]])

    rng.shuffle(sample)
    return sample[:sample_size]


def write_tsv(path: Path, annotations: list[Annotation]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for annotation in annotations:
            f.write(f"{annotation.surface}\t{' '.join(annotation.morphs)}\n")


def load_tsv_annotations(path: Path) -> dict[str, list[tuple[str, ...]]]:
    annotations: dict[str, list[tuple[str, ...]]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            surface, morph_text = line.rstrip("\n").split("\t", 1)
            annotations[surface] = [tuple(morph_text.split())]
    return annotations


def validate_tsv(path: Path) -> dict[str, int | bool]:
    annotations = load_tsv_annotations(path)
    one_part = sum(1 for choices in annotations.values() if len(choices[0]) == 1)
    two_part = sum(1 for choices in annotations.values() if len(choices[0]) == 2)

    import morfessor

    model = morfessor.BaselineModel()
    model.load_data((1, surface) for surface in annotations)
    model.set_annotations(annotations)
    model.train_batch(max_epochs=1)

    return {
        "rows": sum(1 for _ in open(path, "r", encoding="utf-8")),
        "unique_surfaces": len(annotations),
        "one_part": one_part,
        "two_part": two_part,
        "morfessor_smoke_ok": True,
    }


def print_validation(path: Path) -> None:
    stats = validate_tsv(path)
    print(f"Validated: {path}")
    print(f"  rows: {stats['rows']}")
    print(f"  unique surfaces: {stats['unique_surfaces']}")
    print(f"  one-part segmentations: {stats['one_part']}")
    print(f"  two-part segmentations: {stats['two_part']}")
    print(f"  morfessor smoke ok: {stats['morfessor_smoke_ok']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract DSB metadix surface-form morphology annotations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dix", type=Path, default=Path("apertium-dsb.dsb.metadix"))
    parser.add_argument(
        "--full-output",
        type=Path,
        default=Path("data/processed/dsb/metadix_morph_annotations_full.tsv"),
    )
    parser.add_argument(
        "--sample-output",
        type=Path,
        default=Path("data/processed/dsb/metadix_morph_annotations_1000.tsv"),
    )
    parser.add_argument("--sample-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validate", type=Path, help="Validate an existing TSV and exit.")
    args = parser.parse_args()

    if args.validate:
        print_validation(args.validate)
        return

    annotations = extract_annotations(args.dix)
    sample = balanced_sample(annotations, args.sample_size, args.seed)

    write_tsv(args.full_output, annotations)
    write_tsv(args.sample_output, sample)

    print(f"Wrote full annotations: {args.full_output} ({len(annotations)} rows)")
    print(f"Wrote sampled annotations: {args.sample_output} ({len(sample)} rows)")
    print_validation(args.sample_output)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
