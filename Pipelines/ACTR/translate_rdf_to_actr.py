import json
import re
from pathlib import Path
from typing import TypedDict

from Pipelines.parse_ontology import parse_ontology


class ActrMetadata(TypedDict):
    source: str
    format: str
    chunk_count: int


class ActrChunk(TypedDict):
    name: str
    isa: str
    slots: dict[str, object]


class ActrMemory(TypedDict):
    metadata: ActrMetadata
    chunks: list[ActrChunk]


def _build_class_hierarchy(entities: dict[str, dict[str, list[str]]]) -> dict[str, list[str]]:
    class_hierarchy: dict[str, list[str]] = {}

    for entity, attributes in entities.items():
        if "type" in attributes and "class" in attributes["type"]:
            class_hierarchy[entity] = list(attributes.get("subclassof", []))

    return class_hierarchy


def _get_superclasses(class_hierarchy: dict[str, list[str]], class_name: str) -> list[str]:
    parents = class_hierarchy.get(class_name, [])
    lineage = set(parents)

    for parent in parents:
        lineage.update(_get_superclasses(class_hierarchy, parent))

    return sorted(lineage)


def translate_rdf_to_actr(rewrite: bool = False):
    actr_file_path = Path(__file__).resolve().parent / "cognitive-robotics.actr.json"
    if actr_file_path.exists() and not rewrite:
        return

    entities = parse_ontology()

    class_hierarchy = _build_class_hierarchy(entities)

    memory: ActrMemory = {
        "metadata": {
            "source": "cognitive-robotics.rdf",
            "format": "act-r-inspired-declarative-memory",
            "chunk_count": 0,
        },
        "chunks": [],
    }

    chunks: list[ActrChunk] = []

    grouped_entities: dict[str, list[str]] = {}

    for entity, attributes in entities.items():
        if not attributes or entity == "cognitive-robotics":
            continue

        primary_type = "other"
        if "type" in attributes:
            specific_types = [t for t in attributes["type"] if t != "namedindividual"]
            if specific_types:
                primary_type = specific_types[0]

        if primary_type not in grouped_entities:
            grouped_entities[primary_type] = []
        grouped_entities[primary_type].append(entity)

    for group_type in sorted(grouped_entities.keys()):
        for entity in sorted(grouped_entities[group_type]):
            attributes = entities[entity]

            raw_types: list[str] = []
            if "type" in attributes:
                raw_types = sorted([t for t in attributes["type"] if t != "namedindividual"])

            all_types = set(raw_types)
            for raw_type in raw_types:
                all_types.update(_get_superclasses(class_hierarchy, raw_type))

            chunk: ActrChunk = {
                "name": entity,
                "isa": group_type,
                "slots": {
                    "id": entity,
                },
            }

            chunk_slots = chunk["slots"]
            assert isinstance(chunk_slots, dict)

            if raw_types:
                chunk_slots["type"] = raw_types
                chunk_slots["all_types"] = sorted(all_types)

            for attr_key in sorted(attributes.keys()):
                if attr_key == "type":
                    continue

                normalized_values: list[object] = []
                for attr_val in sorted(attributes[attr_key]):
                    if re.fullmatch(r"-?\d+", attr_val):
                        normalized_values.append(int(attr_val))
                    else:
                        normalized_values.append(attr_val)

                if len(normalized_values) == 1:
                    chunk_slots[attr_key] = normalized_values[0]
                else:
                    chunk_slots[attr_key] = normalized_values

            chunks.append(chunk)

    memory["metadata"]["chunk_count"] = len(chunks)
    memory["chunks"] = chunks

    with open(actr_file_path, "w", encoding="utf-8") as f:
        json.dump(memory, f, indent=2)
        f.write("\n")
