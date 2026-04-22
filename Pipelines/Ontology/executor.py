import copy
from rdflib import URIRef
from typing import Optional

from Pipelines.Ontology.world_model import OntologyWorld, uri_fragment
from Pipelines.parse_language import ActionObject, ActionSequence


class OntologyExecutor:

    def __init__(self, world: OntologyWorld):
        self.world = world
        self.trace: list[str] = []
        self.failure_reason: Optional[str] = None

    def _execute_get(self, im: ActionObject) -> None:
        w = self.world
        target = self._resolve_target_object(im)
        if target is None:
            return

        if w.is_static_object(target):
            self._fail("non-manipulable")
            return

        if w.agent_holding_object() is not None:
            self._fail("capacity-limit")
            return

        target_loc = w.object_at_location(target)
        if target_loc is None:
            self._fail("unresolved-target")
            return

        agent_loc = w.agent_location()
        if agent_loc != target_loc:
            if not self._move_agent_to(target_loc):
                return

        w.clear_object_location(target)
        w.set_agent_holding(target)
        self.trace.append(f"[ACTION] Agent got {uri_fragment(target)} at {uri_fragment(target_loc)}")

    def _execute_query_boolean(self, im: ActionObject) -> None:
        w = self.world
        target = self._resolve_target_object(im)
        if target is None:
            self._fail("unresolved-target")
            return

        if "destination-class" not in im and "desired-x" not in im:
            grounded = w.object_at_location(target) is not None
            truth = "TRUE" if grounded else "FALSE"
            self.trace.append(f"[QUERY] Is {uri_fragment(target)} grounded somewhere? {truth}")
            return

        dest_loc = self._resolve_destination_location(im)
        if self.failure_reason or dest_loc is None:
            return

        target_loc = w.object_at_location(target)
        truth_value = target_loc == dest_loc
        dest_class = str(im.get("destination-class", "location")).lower()
        verdict = "TRUE" if truth_value else "FALSE"
        self.trace.append(f"[QUERY] Is {uri_fragment(target)} at/in {dest_class}? {verdict}")

    def _execute_query_inventory(self) -> None:
        held = self.world.agent_holding_object()
        if held is None:
            self.trace.append("[QUERY] agent_01 is holding nothing")
            return
        self.trace.append(f"[QUERY] agent_01 is holding {uri_fragment(held)}")

    def _execute_query_location(self, im: ActionObject) -> None:
        target = self._resolve_target_object(im)
        if target is None:
            return

        loc = self.world.object_at_location(target)
        if loc is None:
            self.trace.append(f"[QUERY] Location of {uri_fragment(target)} is UNKNOWN")
            return

        self.trace.append(f"[QUERY] {uri_fragment(target)} is at {uri_fragment(loc)}")

    def _execute_set(self, im: ActionObject) -> None:
        w = self.world

        if im.get("target-class"):
            target = self._resolve_target_object(im)
        else:
            target = w.agent_holding_object()

        if target is None:
            self._fail("not-holding")
            return

        if w.is_static_object(target):
            self._fail("non-manipulable")
            return

        held = w.agent_holding_object()
        if held != target:
            self._fail("not-holding")
            return

        dest_loc = self._resolve_destination_location(im)
        if self.failure_reason:
            return

        if dest_loc is not None:
            agent_loc = w.agent_location()
            if agent_loc != dest_loc:
                if not self._move_agent_to(dest_loc):
                    return

        current = w.agent_location()
        if current is None:
            self._fail("unresolved-destination")
            return

        w.set_agent_holding(None)
        w.set_object_location(target, current)
        self.trace.append(f"[ACTION] Agent set {uri_fragment(target)} at {uri_fragment(current)}")

    def _fail(self, reason: str) -> None:
        self.failure_reason = reason

    def _log_move(self, old_loc: Optional[URIRef], new_loc: URIRef) -> None:
        old_f = uri_fragment(old_loc) if old_loc else "none"
        new_f = uri_fragment(new_loc)
        self.trace.append(f"[ACTION] Agent moved from {old_f} to {new_f}")

    def _move_agent_to(self, dest_loc: URIRef) -> bool:
        coords = self.world.get_location_coords(dest_loc)
        if coords is None:
            self._fail("out-of-bounds")
            return False
        old = self.world.agent_location()
        self.world.set_agent_location(dest_loc)
        self._log_move(old, dest_loc)
        return True

    def _resolve_destination_location(self, im: ActionObject) -> Optional[URIRef]:
        if "desired-x" in im and "desired-y" in im:
            loc = self.world.location_by_coords(int(im["desired-x"]), int(im["desired-y"]))
            if loc is None:
                self._fail("out-of-bounds")
                return None
            return loc

        dc = im.get("destination-class")
        if dc:
            ent = self.world.retrieve_entity(str(dc), _modifiers(im, "destination-modifiers"))
            if ent is None:
                self._fail("unknown-destination")
                return None
            loc = self.world.object_at_location(ent)
            if loc is None:
                self._fail("unresolved-destination")
                return None
            return loc

        return None

    def _resolve_move_destination(self, im: ActionObject) -> Optional[URIRef]:
        w = self.world
        if "desired-x" in im and "desired-y" in im:
            loc = w.location_by_coords(int(im["desired-x"]), int(im["desired-y"]))
            if loc is None:
                self._fail("out-of-bounds")
                return None
            return loc

        if "direction" in im:
            cur = w.agent_location()
            if cur is None:
                self._fail("unresolved-destination")
                return None
            coords = w.get_location_coords(cur)
            if coords is None:
                self._fail("unresolved-destination")
                return None
            cx, cy = coords
            direction = str(im["direction"]).lower()
            delta_map = {"north": (0, 1), "east": (1, 0), "south": (0, -1), "west": (-1, 0)}
            if direction not in delta_map:
                self._fail("unresolved-destination")
                return None
            dist = int(im.get("distance", 1))
            dx, dy = delta_map[direction]
            nx, ny = cx + dx * dist, cy + dy * dist
            loc = w.location_by_coords(nx, ny)
            if loc is None:
                self._fail("out-of-bounds")
                return None
            return loc

        if "destination-class" in im:
            ent = w.retrieve_entity(str(im["destination-class"]), _modifiers(im, "destination-modifiers"))
            if ent is None:
                self._fail("unknown-destination")
                return None
            loc = w.object_at_location(ent)
            if loc is None:
                self._fail("unresolved-destination")
                return None
            return loc

        if "target-class" in im:
            ent = w.retrieve_entity(str(im["target-class"]), _modifiers(im, "target-modifiers"))
            if ent is None:
                self._fail("unresolved-target")
                return None
            loc = w.object_at_location(ent)
            if loc is None:
                self._fail("unresolved-target")
                return None
            return loc

        if im.get("distance") is not None and "direction" not in im:
            self._fail("unresolved-destination")
            return None

        self._fail("unresolved-destination")
        return None

    def _resolve_target_object(self, im: ActionObject) -> Optional[URIRef]:
        tc = im.get("target-class")
        if not tc:
            self._fail("missing-target-key")    # parser omitted the required slot
            return None
        entity = self.world.retrieve_entity(str(tc), _modifiers(im, "target-modifiers"))
        if entity is None:
            self._fail("unresolved-target")     # slot present, no matching individual
        return entity

    def _run_one(self, im: ActionObject) -> None:
        action = im.get("action")
        if action == "move":
            dest = self._resolve_move_destination(im)
            if dest is None or self.failure_reason:
                return
            if not self._move_agent_to(dest):
                return
            return

        if action == "get":
            self._execute_get(im)
            return

        if action == "set":
            self._execute_set(im)
            return

        if action == "query-location":
            self._execute_query_location(im)
            return

        if action == "query-inventory":
            self._execute_query_inventory()
            return

        if action == "query-boolean":
            self._execute_query_boolean(im)
            return

        self._fail("unknown-action")

    def run(self, commands: ActionSequence) -> str:
        self.trace = []
        self.failure_reason = None

        for index, command in enumerate(commands, start=1):
            imaginal = copy.deepcopy(command)
            self._run_one(imaginal)
            if self.failure_reason is not None:
                self.trace.append(f"[FAILURE] Command rejected by constraint: {self.failure_reason}")
                break
            if index < len(commands):
                self.trace.append("[SYSTEM] Command finished. Transitioning to next command...")

        if self.failure_reason is None:
            self.trace.append("[SUCCESS] Command sequence execution complete. Agent is idle.")

        return "\n".join(self.trace)


def _modifiers(cmd: ActionObject, key: str) -> list[str]:
    raw = cmd.get(key)
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw.lower()]
    return [str(x).lower() for x in raw]


def execute_action_sequence(world: OntologyWorld, actions: ActionSequence) -> tuple[str, Optional[str]]:
    ex = OntologyExecutor(world)
    trace = ex.run(actions)
    return trace, ex.failure_reason
