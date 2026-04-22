import re
from pathlib import Path

from Pipelines.parse_ontology import parse_ontology


def translate_rdf_to_soar(rewrite: bool = False):
    soar_file_path = Path(__file__).resolve().parent / "rules" / "cognitive-robotics.soar"
    if soar_file_path.exists() and not rewrite:
        return

    entities = parse_ontology()

    rules: list[str] = []

    struct_rule = "sp {elaborate*state*structure\n"
    struct_rule += "    (state <s> ^superstate nil)\n"
    struct_rule += "-->\n"
    struct_rule += "    (<s> ^environment <env>)\n"
    struct_rule += "}\n"
    rules.append(struct_rule)

    init_prop = "sp {propose*init-environment\n"
    init_prop += "    (state <s> ^superstate nil)\n"
    init_prop += "    -(<s> ^name cognitive-robotics)\n"
    init_prop += "-->\n"
    init_prop += "    (<s> ^operator <o> + >)\n"
    init_prop += "    (<o> ^name init-environment)\n"
    init_prop += "}\n"
    rules.append(init_prop)

    init_base = "sp {apply*init-environment*base\n"
    init_base += "    (state <s> ^operator.name init-environment)\n"
    init_base += "-->\n"
    init_base += "    (<s> ^name cognitive-robotics)\n"
    init_base += "}\n"
    rules.append(init_base)

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
        rules.append(f"\n# {'='*50}\n# ONTOLOGY CATEGORY: {group_type.upper()}\n# {'='*50}")
        
        for entity in sorted(grouped_entities[group_type]):
            attributes = entities[entity]
            safe_entity_name = re.sub(r"[^a-z0-9_]", "_", entity)
            
            entity_rule = f"sp {{apply*init-environment*{safe_entity_name}\n"
            entity_rule += "    (state <s> ^operator.name init-environment\n"
            entity_rule += "        ^environment <env>)\n"
            entity_rule += "-->\n"
            entity_rule += "    (<env> ^entity <e>)\n"
            entity_rule += f"    (<e> ^id {entity}\n"
            
            for attr_key in sorted(attributes.keys()):
                for attr_val in sorted(attributes[attr_key]):
                    entity_rule += f"        ^{attr_key} {attr_val}\n"
                    
            entity_rule += "    )\n"
            entity_rule += "}"
            rules.append(entity_rule)

    soar_code = "\n\n".join(rules) + "\n"

    with open(soar_file_path, "w", encoding="utf-8") as f:
        f.write(soar_code)
