import rdflib
import re
import sys



ONTOLOGY_NAMESPACE = rdflib.Namespace("https://www.upc.edu/yaacoub/ontologies/cognitive-robotics#")



def parse_ontology(rdf_file_path: str) -> dict[str, dict[str, list[str]]]:
    graph = rdflib.Graph()
    graph.parse(rdf_file_path)

    entities: dict[str, dict[str, list[str]]] = {}

    for subj, pred, obj in graph:
        if str(ONTOLOGY_NAMESPACE) not in str(subj):
            continue

        subject_name = str(subj).replace(str(ONTOLOGY_NAMESPACE), "").lower()
        predicate_name = str(pred).split("#")[-1].lower()

        if isinstance(obj, rdflib.URIRef):
            object_val = str(obj).split("#")[-1].split("/")[-1].lower()
        else:
            object_val = str(obj).lower()

        if subject_name not in entities:
            entities[subject_name] = {}

        if predicate_name not in entities[subject_name]:
            entities[subject_name][predicate_name] = []

        if object_val not in entities[subject_name][predicate_name]:
            entities[subject_name][predicate_name].append(object_val)

    return entities



def translate_rdf_to_soar(rdf_file_path: str, soar_file_path: str):
    entities = parse_ontology(rdf_file_path)

    rules: list[str] = []

    # Elaborate the environment container safely
    struct_rule = "sp {elaborate*state*structure\n"
    struct_rule += "    (state <s> ^superstate nil)\n"
    struct_rule += "-->\n"
    struct_rule += "    (<s> ^environment <env>)\n"
    struct_rule += "}\n"
    rules.append(struct_rule)

    # Propose the Initialization Operator
    init_prop = "sp {propose*init-environment\n"
    init_prop += "    (state <s> ^superstate nil)\n"
    init_prop += "    -(<s> ^name cognitive-robotics)\n"
    init_prop += "-->\n"
    init_prop += "    (<s> ^operator <o> + >)\n"
    init_prop += "    (<o> ^name init-environment)\n"
    init_prop += "}\n"
    rules.append(init_prop)

    # Set the state name (retract the operator at the end)
    init_base = "sp {apply*init-environment*base\n"
    init_base += "    (state <s> ^operator.name init-environment)\n"
    init_base += "-->\n"
    init_base += "    (<s> ^name cognitive-robotics)\n"
    init_base += "}\n"
    rules.append(init_base)

    # Group entities dynamically for clean Soar code readability
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

    # Populate Entities dynamically
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

    with open(soar_file_path, "w") as f:
        f.write(soar_code)



if __name__ == "__main__":
    
    if len(sys.argv) != 3:
        print("Usage: python rdf_to_soar.py <rdf_file> <soar_file>")
        sys.exit(1)
    
    rdf_file = sys.argv[1]
    soar_file = sys.argv[2]
    translate_rdf_to_soar(rdf_file, soar_file)