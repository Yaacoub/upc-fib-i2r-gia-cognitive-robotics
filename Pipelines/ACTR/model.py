import copy
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional

ActionObject = dict[str, Any]
ActionSequence = list[ActionObject]


@dataclass
class Chunk:
    name: str
    isa: str
    slots: dict[str, Any]

    def has_value(self, slot_name: str, expected_value: Any) -> bool:
        return expected_value in self.slot_values(slot_name)

    def slot_values(self, slot_name: str) -> list[Any]:
        value = self.slots.get(slot_name)
        if value is None:
            return []
        elif isinstance(value, list):
            return value  # type: ignore
        return [value]


@dataclass
class Buffers:
    goal: dict[str, Any] = field(default_factory=lambda: {})
    imaginal: dict[str, Any] = field(default_factory=lambda: {})
    retrieval: Optional[Chunk] = None
    manual: Optional[dict[str, Any]] = None


class DeclarativeMemory:
    def __init__(self, chunks: Iterable[Chunk]):
        self.chunks: dict[str, Chunk] = {chunk.name: chunk for chunk in chunks}
        self.access_count: dict[str, int] = {}

    def add_chunk(self, chunk: Chunk) -> None:
        self.chunks[chunk.name] = chunk

    def get(self, chunk_name: str) -> Optional[Chunk]:
        chunk = self.chunks.get(chunk_name)
        if chunk is not None:
            self.record_access(chunk.name)
        return chunk

    def iter_chunks(self) -> Iterable[Chunk]:
        return self.chunks.values()

    def record_access(self, chunk_name: str) -> None:
        self.access_count[chunk_name] = self.access_count.get(chunk_name, 0) + 1


class ActRModel:
    def __init__(self, chunks: Iterable[Chunk], cycle_seconds: float = 0.05):
        self.memory = DeclarativeMemory(chunks)
        self.buffers = Buffers()
        self.cycle_seconds = cycle_seconds
        self.cycle_count = 0
        self.elapsed_seconds = 0.0
        self.trace: list[str] = []
        self.failure_reason: Optional[str] = None

        for skill_chunk in self._skill_chunks():
            self.memory.add_chunk(skill_chunk)

    def _agent_chunk(self) -> Optional[Chunk]:
        for chunk in self.memory.iter_chunks():
            if self._chunk_has_type(chunk, "agent") and chunk.name == "agent_01":
                return chunk
        return None

    def _agent_holding(self) -> Optional[str]:
        agent = self._agent_chunk()
        return agent.slots.get("isholding") if agent else None

    def _agent_location(self) -> Optional[str]:
        agent = self._agent_chunk()
        return agent.slots.get("isatlocation") if agent else None

    def _chunk_has_type(self, chunk: Chunk, type_name: str) -> bool:
        return type_name in chunk.slot_values("all_types") or type_name in chunk.slot_values("type") or chunk.isa == type_name

    def _command_has_explicit_target(self) -> bool:
        return "target-class" in self.buffers.imaginal or "target-object" in self.buffers.imaginal

    def _destination_resolution_complete(self) -> bool:
        action = self.buffers.imaginal.get("action")
        if action == "move":
            return "destination" in self.buffers.imaginal

        if action == "set":
            if "destination-class" not in self.buffers.imaginal and "desired-x" not in self.buffers.imaginal:
                return True
            return "destination" in self.buffers.imaginal

        if action == "query-boolean":
            if "destination-class" not in self.buffers.imaginal and "desired-x" not in self.buffers.imaginal:
                return True
            return "destination" in self.buffers.imaginal

        return True

    def _drop_object(self, target_name: Optional[str]) -> None:
        if not target_name:
            self._set_failure("unresolved-target")
            return

        agent = self._agent_chunk()
        target = self._get_world_chunk(target_name)
        if agent is None or target is None:
            self._set_failure("unresolved-target")
            return

        current_location = agent.slots.get("isatlocation")
        if agent.slots.get("isholding") == target_name:
            agent.slots.pop("isholding", None)
        target.slots["isatlocation"] = current_location
        self.trace.append(f"[ACTION] Agent set {target_name} at {current_location}")

    def _execute_manual_action(self) -> None:
        if self.buffers.manual is None:
            self._set_failure("stuck")
            return

        operator = self.buffers.manual.get("operator")

        if operator == "move":
            destination = self.buffers.manual.get("destination")
            if destination is None:
                self._set_failure("unresolved-destination")
                return
            self._log_production("manual>execute-move")
            self._move_agent(destination)
            if self.buffers.imaginal.get("action") == "move":
                self.buffers.goal["status"] = "completed"
            self.buffers.manual = None
            return

        if operator == "get":
            target_name = self.buffers.manual.get("target")
            self._log_production("manual>execute-get")
            self._pick_up_object(target_name)
            self.buffers.goal["status"] = "completed"
            self.buffers.manual = None
            return

        if operator == "set":
            target_name = self.buffers.manual.get("target")
            self._log_production("manual>execute-set")
            self._drop_object(target_name)
            self.buffers.goal["status"] = "completed"
            self.buffers.manual = None
            return

        if operator == "query-location":
            target_name = self.buffers.manual.get("target")
            self._log_production("manual>execute-query-location")
            self._query_location(target_name)
            self.buffers.goal["status"] = "completed"
            self.buffers.manual = None
            return

        if operator == "query-inventory":
            self._log_production("manual>execute-query-inventory")
            self._query_inventory()
            self.buffers.goal["status"] = "completed"
            self.buffers.manual = None
            return

        if operator == "query-boolean":
            self._log_production("manual>execute-query-boolean")
            self._query_boolean(
                self.buffers.manual.get("target"),
                self.buffers.manual.get("destination"),
            )
            self.buffers.goal["status"] = "completed"
            self.buffers.manual = None
            return

        self._set_failure("stuck")

    def _fire_next_production(self) -> None:
        phase = self.buffers.goal.get("phase")

        if phase == "encode":
            self._log_production("goal>encode-command")
            self.buffers.goal["phase"] = "retrieve-skill"
            return

        if phase == "retrieve-skill":
            self._retrieve_skill_chunk()
            return

        if phase == "resolve-target":
            if self._target_resolution_complete():
                self._log_production("imaginal>target-ready")
                self.buffers.goal["phase"] = "resolve-destination"
                return

            self._resolve_target()
            return

        if phase == "resolve-destination":
            if self._destination_resolution_complete():
                self._log_production("imaginal>destination-ready")
                self.buffers.goal["phase"] = "execute"
                return

            self._resolve_destination()
            return

        if phase == "execute":
            if self.buffers.manual is None:
                self._schedule_manual_action()
                return

            self._execute_manual_action()
            return

        self._set_failure("stuck")

    def _get_world_chunk(self, chunk_name: Optional[str]) -> Optional[Chunk]:
        if not chunk_name:
            return None
        return self.memory.chunks.get(chunk_name)

    def _log_production(self, production_name: str) -> None:
        self.cycle_count += 1
        self.elapsed_seconds = self.cycle_count * self.cycle_seconds
        self.trace.append(f"[CYCLE {self.cycle_count:03d} | {self.elapsed_seconds:.2f}s] {production_name}")

    def _move_agent(self, destination_name: str) -> None:
        agent = self._agent_chunk()
        if agent is None:
            self._set_failure("unresolved-agent")
            return

        old_location = agent.slots.get("isatlocation")
        agent.slots["isatlocation"] = destination_name
        self.trace.append(f"[ACTION] Agent moved from {old_location} to {destination_name}")

    def _pick_up_object(self, target_name: Optional[str]) -> None:
        if not target_name:
            self._set_failure("unresolved-target")
            return

        agent = self._agent_chunk()
        target = self._get_world_chunk(target_name)
        if agent is None or target is None:
            self._set_failure("unresolved-target")
            return

        target_location = target.slots.pop("isatlocation", None)
        agent.slots["isholding"] = target_name
        self.trace.append(f"[ACTION] Agent got {target_name} at {target_location}")

    def _prime_buffers(self, command: ActionObject, command_index: int) -> None:
        self.buffers = Buffers(
            goal={
                "phase": "encode",
                "status": "pending",
                "command-index": command_index,
                "action": command.get("action"),
            },
            imaginal=copy.deepcopy(command),
        )

    def _query_boolean(self, target_name: Optional[str], destination_name: Optional[str]) -> None:
        target = self._get_world_chunk(target_name)
        if target is None:
            self._set_failure("unresolved-target")
            return

        if destination_name is None:
            truth_value = "true" if target.slots.get("isatlocation") is not None else "false"
            self.trace.append(f"[QUERY] Is {target.name} grounded somewhere? {truth_value.upper()}")
            return

        target_location = target.slots.get("isatlocation")
        truth_value = target_location == destination_name
        destination_class = self.buffers.imaginal.get("destination-class", "location")
        verdict = "TRUE" if truth_value else "FALSE"
        self.trace.append(f"[QUERY] Is {target.name} at/in {destination_class}? {verdict}")

    def _query_inventory(self) -> None:
        held_object = self._agent_holding()
        if held_object is None:
            self.trace.append("[QUERY] agent_01 is holding nothing")
            return

        self.trace.append(f"[QUERY] agent_01 is holding {held_object}")

    def _query_location(self, target_name: Optional[str]) -> None:
        target = self._get_world_chunk(target_name)
        if target is None:
            self._set_failure("unresolved-target")
            return

        location_name = target.slots.get("isatlocation")
        if location_name is None:
            self.trace.append(f"[QUERY] Location of {target.name} is UNKNOWN")
            return

        self.trace.append(f"[QUERY] {target.name} is at {location_name}")

    def _resolve_destination(self) -> None:
        action = self.buffers.imaginal.get("action")

        if action == "move" and "target-location" in self.buffers.imaginal:
            self.buffers.imaginal["destination"] = self.buffers.imaginal["target-location"]
            self._log_production("imaginal>target-location-to-destination")
            return

        if action == "move" and "direction" in self.buffers.imaginal and "desired-x" not in self.buffers.imaginal:
            self.buffers.goal["phase"] = "execute"
            self._log_production("imaginal>direction-awaiting-computation")
            return

        if "desired-x" in self.buffers.imaginal and "desired-y" in self.buffers.imaginal:
            location_chunk = self._retrieve_location_by_coordinates(
                int(self.buffers.imaginal["desired-x"]),
                int(self.buffers.imaginal["desired-y"]),
            )
            self._log_production("retrieval>location-by-coordinates")
            if location_chunk is None:
                self._set_failure("out-of-bounds")
                return
            self.buffers.imaginal["destination"] = location_chunk.name
            self._log_production("imaginal>bind-coordinate-destination")
            return

        destination_class = self.buffers.imaginal.get("destination-class")
        destination_modifiers = self.buffers.imaginal.get("destination-modifiers", [])
        if not destination_class:
            self._set_failure("unknown-destination")
            return

        destination_chunk = self._retrieve_entity(destination_class, destination_modifiers)
        self._log_production("retrieval>destination-by-features")
        if destination_chunk is None:
            self._set_failure("unknown-destination")
            return

        destination_location = destination_chunk.slots.get("isatlocation")
        if destination_location is None:
            self._set_failure("unresolved-destination")
            return

        self.buffers.retrieval = destination_chunk
        self.buffers.imaginal["destination"] = destination_location
        self._log_production("imaginal>bind-destination")

    def _resolve_move_destination(self) -> Optional[str]:
        if "destination" in self.buffers.imaginal:
            return self.buffers.imaginal["destination"]

        direction = self.buffers.imaginal.get("direction")
        if not direction:
            self._set_failure("unresolved-destination")
            return None

        delta_map = {
            "north": (0, 1),
            "east": (1, 0),
            "south": (0, -1),
            "west": (-1, 0),
        }

        current_location_chunk = self._get_world_chunk(self._agent_location())
        if current_location_chunk is None:
            self._set_failure("unresolved-destination")
            return None

        distance = int(self.buffers.imaginal.get("distance", 1))
        delta_x, delta_y = delta_map[direction]
        desired_x = int(current_location_chunk.slots["hasxcoordinate"]) + (delta_x * distance)
        desired_y = int(current_location_chunk.slots["hasycoordinate"]) + (delta_y * distance)
        self.buffers.imaginal["desired-x"] = desired_x
        self.buffers.imaginal["desired-y"] = desired_y
        self.buffers.goal["phase"] = "resolve-destination"
        self.buffers.manual = None
        self._log_production("imaginal>compute-directional-destination")
        return None

    def _resolve_target(self) -> None:
        action = self.buffers.imaginal.get("action")

        if action == "set" and not self._command_has_explicit_target():
            held_object = self._agent_holding()
            self._log_production("retrieval>held-object")
            if held_object is None:
                self._set_failure("not-holding")
                return
            self.buffers.imaginal["target-object"] = held_object
            self._log_production("imaginal>infer-drop-target")
            return

        target_class = self.buffers.imaginal.get("target-class")
        target_modifiers = self.buffers.imaginal.get("target-modifiers", [])
        if not target_class:
            self._set_failure("missing-target-key")   # slot absent from action dict
            return

        target_chunk = self._retrieve_entity(target_class, target_modifiers)
        if target_chunk is None:
            self._set_failure("unresolved-target")    # slot present, no matching chunk
            return

        self.buffers.retrieval = target_chunk
        self.buffers.imaginal["target-object"] = target_chunk.name
        location_name = target_chunk.slots.get("isatlocation")
        if location_name is not None:
            self.buffers.imaginal["target-location"] = location_name
        self._log_production("imaginal>bind-target")

    def _retrieve_best_chunk(self, predicate: Callable[[Chunk], bool]) -> Optional[Chunk]:
        candidates = [chunk for chunk in self.memory.iter_chunks() if predicate(chunk)]
        if not candidates:
            return None

        candidates.sort(key=lambda chunk: (self.memory.access_count.get(chunk.name, 0), chunk.name), reverse=True)
        best = candidates[0]
        self.memory.record_access(best.name)
        return best

    def _retrieve_entity(self, entity_class: str, modifiers: list[str]) -> Optional[Chunk]:
        agent_holding = self._agent_holding()
        candidates: list[tuple[float, Chunk]] = []

        for chunk in self.memory.iter_chunks():
            if chunk.isa == "skill" or chunk.isa == "class":
                continue

            if not self._chunk_has_type(chunk, entity_class):
                continue

            score = 10.0
            
            for modifier in modifiers:
                matched = False
                for slot_value in chunk.slots.values():
                    if isinstance(slot_value, list) and modifier in slot_value:
                        matched = True
                        break
                    elif slot_value == modifier:
                        matched = True
                        break
                
                if matched or modifier in chunk.name:
                    score += 20.0
                else:
                    score -= 2.0

            score += self.memory.access_count.get(chunk.name, 0) * 0.05
            if agent_holding and chunk.name == agent_holding:
                score += 5.0

            candidates.append((score, chunk))

        if not candidates:
            return None

        candidates.sort(key=lambda item: (-item[0], item[1].name))
        best_chunk = candidates[0][1]
        self.memory.record_access(best_chunk.name)
        return best_chunk

    def _retrieve_location_by_coordinates(self, x_value: int, y_value: int) -> Optional[Chunk]:
        return self._retrieve_best_chunk(
            lambda chunk: self._chunk_has_type(chunk, "gridlocation")
            and chunk.slots.get("hasxcoordinate") == x_value
            and chunk.slots.get("hasycoordinate") == y_value
        )

    def _retrieve_skill_chunk(self) -> None:
        action = self.buffers.imaginal.get("action")
        skill = self._retrieve_best_chunk(
            lambda chunk: chunk.isa == "skill" and chunk.slots.get("action") == action
        )

        self._log_production("retrieval>skill-schema")
        if skill is None:
            self._set_failure("unknown-action")
            return

        self.buffers.retrieval = skill
        self.buffers.imaginal["skill-chunk"] = skill.name
        self.buffers.goal["phase"] = "resolve-target"
        self._log_production("imaginal>bind-skill-schema")

    def _schedule_get(self) -> None:
        target_name = self.buffers.imaginal.get("target-object")
        target_chunk = self._get_world_chunk(target_name)
        if target_chunk is None:
            self._set_failure("unresolved-target")
            return

        if self._chunk_has_type(target_chunk, "staticobject"):
            self._set_failure("non-manipulable")
            return

        if self._agent_holding() is not None:
            self._set_failure("capacity-limit")
            return

        target_location = target_chunk.slots.get("isatlocation")
        agent_location = self._agent_location()
        if target_location is None:
            self._set_failure("unresolved-target")
            return

        if agent_location != target_location:
            self.buffers.manual = {"operator": "move", "destination": target_location}
            self._log_production("manual>schedule-move-to-target")
            return

        self.buffers.manual = {"operator": "get", "target": target_name}
        self._log_production("manual>schedule-get")

    def _schedule_manual_action(self) -> None:
        action = self.buffers.imaginal.get("action")

        if action == "move":
            destination = self._resolve_move_destination()
            if destination is None:
                return
            self.buffers.manual = {"operator": "move", "destination": destination}
            self._log_production("manual>schedule-move")
            return

        if action == "get":
            self._schedule_get()
            return

        if action == "set":
            self._schedule_set()
            return

        if action == "query-location":
            self.buffers.manual = {
                "operator": "query-location",
                "target": self.buffers.imaginal.get("target-object"),
            }
            self._log_production("manual>schedule-query-location")
            return

        if action == "query-inventory":
            self.buffers.manual = {"operator": "query-inventory"}
            self._log_production("manual>schedule-query-inventory")
            return

        if action == "query-boolean":
            self.buffers.manual = {
                "operator": "query-boolean",
                "target": self.buffers.imaginal.get("target-object"),
                "destination": self.buffers.imaginal.get("destination"),
            }
            self._log_production("manual>schedule-query-boolean")
            return

        self._set_failure("unknown-action")

    def _schedule_set(self) -> None:
        target_name = self.buffers.imaginal.get("target-object")
        if not target_name:
            self._set_failure("not-holding")
            return

        target_chunk = self._get_world_chunk(target_name)
        if target_chunk is None:
            self._set_failure("unresolved-target")
            return

        if self._chunk_has_type(target_chunk, "staticobject"):
            self._set_failure("non-manipulable")
            return

        held_object = self._agent_holding()
        if held_object != target_name:
            self._set_failure("not-holding")
            return

        destination = self.buffers.imaginal.get("destination")
        if destination is not None and self._agent_location() != destination:
            self.buffers.manual = {"operator": "move", "destination": destination}
            self._log_production("manual>schedule-move-to-destination")
            return

        self.buffers.manual = {"operator": "set", "target": target_name}
        self._log_production("manual>schedule-set")

    def _set_failure(self, reason: str) -> None:
        self.failure_reason = reason

    @staticmethod
    def _skill_chunks() -> list[Chunk]:
        return [
            Chunk("skill_move", "skill", {"action": "move"}),
            Chunk("skill_get", "skill", {"action": "get"}),
            Chunk("skill_set", "skill", {"action": "set"}),
            Chunk("skill_query_location", "skill", {"action": "query-location"}),
            Chunk("skill_query_inventory", "skill", {"action": "query-inventory"}),
            Chunk("skill_query_boolean", "skill", {"action": "query-boolean"}),
        ]

    def _target_resolution_complete(self) -> bool:
        action = self.buffers.imaginal.get("action")
        if action == "set":
            return "target-object" in self.buffers.imaginal

        if action in ("move", "get", "query-location", "query-boolean"):
            if action == "move" and "target-class" not in self.buffers.imaginal:
                return True
            return "target-object" in self.buffers.imaginal

        return True

    @classmethod
    def from_chunk_file(cls, chunk_file_path: str) -> "ActRModel":
        with open(chunk_file_path, "r") as handle:
            payload = json.load(handle)

        chunks = [
            Chunk(name=chunk["name"], isa=chunk["isa"], slots=chunk["slots"])
            for chunk in payload.get("chunks", [])
        ]
        return cls(chunks)

    def run(self, commands: ActionSequence, max_cycles_per_command: int = 100) -> str:
        self.trace = []
        self.failure_reason = None

        for index, command in enumerate(commands, start=1):
            self._prime_buffers(command, index)

            for _ in range(max_cycles_per_command):
                if self.failure_reason is not None:
                    break

                if self.buffers.goal.get("status") == "completed":
                    break

                self._fire_next_production()

            if self.failure_reason is not None:
                break

            if self.buffers.goal.get("status") != "completed":
                self._set_failure("stuck")
                break

            if index < len(commands):
                self.trace.append("[SYSTEM] Command finished. Transitioning to next command...")

        if self.failure_reason is None:
            self.trace.append("[SUCCESS] Command sequence execution complete. Agent is idle.")
        else:
            self.trace.append(f"[FAILURE] Command rejected by constraint: {self.failure_reason}")

        return "\n".join(self.trace)
