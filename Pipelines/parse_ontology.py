import rdflib
from pathlib import Path


Subject = str
Predicate = str
Object = str


def parse_ontology() -> dict[Subject, dict[Predicate, list[Object]]]:
    namespace = rdflib.Namespace("https://www.upc.edu/yaacoub/ontologies/cognitive-robotics#")
    rdf_file_path = Path(__file__).resolve().parent.parent / "Domain Modeling" / "cognitive-robotics.rdf"

    graph = rdflib.Graph()
    graph.parse(rdf_file_path)

    entities: dict[Subject, dict[Predicate, list[Object]]] = {}

    for subj, pred, obj in graph:
        if str(namespace) not in str(subj):
            continue

        subject_name = str(subj).replace(str(namespace), "").lower()
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