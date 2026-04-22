import rdflib
from pathlib import Path
from rdflib import Graph, Literal, URIRef
from rdflib.namespace import OWL, RDF, RDFS
from typing import Any, Iterable, Optional


NAMESPACE = rdflib.Namespace("https://www.upc.edu/yaacoub/ontologies/cognitive-robotics#")


class OntologyWorld:

    def __init__(self, graph: Graph):
        self.g = graph
        self._agent = NAMESPACE.Agent_01
        self._superclasses = self._build_superclass_index()

    def _build_superclass_index(self) -> dict[str, set[str]]:
        index: dict[str, set[str]] = {}
        for s, _, o in self.g.triples((None, RDFS.subClassOf, None)):
            if not isinstance(s, URIRef) or not isinstance(o, URIRef):
                continue
            child = uri_fragment(s)
            parent = uri_fragment(o)
            if child not in index:
                index[child] = set()
            index[child].add(parent)
        changed = True
        while changed:
            changed = False
            for child, parents in list(index.items()):
                for p in list(parents):
                    for gp in index.get(p, ()):
                        if gp not in parents:
                            parents.add(gp)
                            changed = True
        return index

    @classmethod
    def from_file(cls, path: Optional[Path] = None) -> OntologyWorld:
        return cls(load_graph(path))

    def copy(self) -> OntologyWorld:
        """Deep copy of the graph for one command execution."""
        other = Graph()
        other += self.g
        return OntologyWorld(other)

    def expanded_types(self, type_fragment: str) -> set[str]:
        out = {type_fragment.lower()}
        stack = [type_fragment.lower()]
        while stack:
            cur = stack.pop()
            for sup in self._superclasses.get(cur, ()):
                if sup not in out:
                    out.add(sup)
                    stack.append(sup)
        return out

    def agent_holding_object(self) -> Optional[URIRef]:
        for _, _, obj in self.g.triples((self._agent, NAMESPACE.isHolding, None)):
            if isinstance(obj, URIRef):
                return obj
        return None

    def agent_location(self) -> Optional[URIRef]:
        for _, _, loc in self.g.triples((self._agent, NAMESPACE.isAtLocation, None)):
            if isinstance(loc, URIRef):
                return loc
        return None

    def clear_object_location(self, obj: URIRef) -> None:
        self.g.remove((obj, NAMESPACE.isAtLocation, None))

    def get_location_coords(self, loc: URIRef) -> Optional[tuple[int, int]]:
        x_val: Optional[int] = None
        y_val: Optional[int] = None
        for _, _, lit in self.g.triples((loc, NAMESPACE.hasXCoordinate, None)):
            try:
                x_val = int(str(lit))
            except (TypeError, ValueError):
                pass
        for _, _, lit in self.g.triples((loc, NAMESPACE.hasYCoordinate, None)):
            try:
                y_val = int(str(lit))
            except (TypeError, ValueError):
                pass
        if x_val is None or y_val is None:
            return None
        return x_val, y_val

    def has_type(self, ind: URIRef, type_name: str) -> bool:
        want = type_name.lower()
        for t in self.object_types(ind):
            if want == t or want in self.expanded_types(t):
                return True
        return False

    def individuals(self) -> Iterable[URIRef]:
        for s in self._named_individual_uris():
            yield s

    def is_grid_location(self, ind: URIRef) -> bool:
        return self.has_type(ind, "gridlocation")

    def is_manipulable_object(self, ind: URIRef) -> bool:
        return self.has_type(ind, "manipulableobject")

    def is_static_object(self, ind: URIRef) -> bool:
        return self.has_type(ind, "staticobject")

    def location_by_coords(self, x: int, y: int) -> Optional[URIRef]:
        for loc in self.individuals():
            if not self.is_grid_location(loc):
                continue
            coords = self.get_location_coords(loc)
            if coords == (x, y):
                return loc
        return None

    def object_at_location(self, obj: URIRef) -> Optional[URIRef]:
        for _, _, loc in self.g.triples((obj, NAMESPACE.isAtLocation, None)):
            if isinstance(loc, URIRef):
                return loc
        return None

    def object_color(self, obj: URIRef) -> Optional[str]:
        for _, _, lit in self.g.triples((obj, NAMESPACE.hasColor, None)):
            return _literal_str(lit)
        return None

    def object_types(self, ind: URIRef) -> set[str]:
        types: set[str] = set()
        for _, _, t in self.g.triples((ind, RDF.type, None)):
            if not isinstance(t, URIRef):
                continue
            frag = uri_fragment(t)
            if frag in ("namedindividual", "class", "ontology"):
                continue
            if str(t) == str(OWL.NamedIndividual):
                continue
            types.add(frag)
        return types

    def retrieve_entity(self, entity_class: str, modifiers: list[str]) -> Optional[URIRef]:
        """Best-matching individual (same scoring idea as ActRModel._retrieve_entity)."""
        agent_hold = self.agent_holding_object()
        want = entity_class.lower()
        candidates: list[tuple[float, URIRef]] = []

        for ind in self.individuals():
            if self.is_grid_location(ind):
                continue
            if ind == self._agent and want != "agent":
                continue
            if not self.has_type(ind, want):
                continue

            score = 10.0
            name = uri_fragment(ind)
            for mod in modifiers:
                mod_s = str(mod).lower()
                matched = False
                col = self.object_color(ind)
                if col and mod_s == col:
                    matched = True
                for t in self.object_types(ind):
                    if mod_s == t:
                        matched = True
                if matched or mod_s in name:
                    score += 20.0
                else:
                    score -= 2.0

            score += 0.05 * (1 if agent_hold == ind else 0)
            candidates.append((score, ind))

        if not candidates:
            return None
        candidates.sort(key=lambda item: (-item[0], uri_fragment(item[1])))
        return candidates[0][1]

    def set_agent_holding(self, obj: Optional[URIRef]) -> None:
        self.g.remove((self._agent, NAMESPACE.isHolding, None))
        if obj is not None:
            self.g.add((self._agent, NAMESPACE.isHolding, obj))

    def set_agent_location(self, loc: URIRef) -> None:
        self.g.remove((self._agent, NAMESPACE.isAtLocation, None))
        self.g.add((self._agent, NAMESPACE.isAtLocation, loc))

    def set_object_location(self, obj: URIRef, loc: URIRef) -> None:
        self.g.remove((obj, NAMESPACE.isAtLocation, None))
        self.g.add((obj, NAMESPACE.isAtLocation, loc))

    def _named_individual_uris(self) -> list[URIRef]:
        out: set[URIRef] = set()
        for s in self.g.subjects(RDF.type):
            if not isinstance(s, URIRef) or not str(s).startswith(str(NAMESPACE)):
                continue
            types = list(self.g.objects(s, RDF.type))
            owl_meta = (OWL.Class, OWL.ObjectProperty, OWL.DatatypeProperty, OWL.Ontology)
            if any(t in owl_meta for t in types):
                continue
            out.add(s)
        return sorted(out, key=lambda u: uri_fragment(u))


def _literal_str(val: Any) -> Optional[str]:
    if val is None:
        return None
    if isinstance(val, Literal):
        return str(val).lower()
    return str(val).lower()


def default_rdf_path() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "Domain Modeling" / "cognitive-robotics.rdf"


def load_graph(path: Optional[Path] = None) -> Graph:
    graph = Graph()
    graph.bind("cr", NAMESPACE)
    graph.parse(path or default_rdf_path())
    return graph


def uri_fragment(node: URIRef) -> str:
    return str(node).split("#")[-1].split("/")[-1].lower()
