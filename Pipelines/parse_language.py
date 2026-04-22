import json
import time
from dataclasses import dataclass
from dotenv import load_dotenv
from google import genai
from pathlib import Path
from typing import Any, cast, Optional, Union


from Pipelines.load_dataset import Command, GroundTruth


ActionObject = dict[str, Any]
ActionSequence = list[ActionObject]
CommandID = str
ParseTime = Optional[float]


@dataclass
class TokenUsage:
    prompt: Optional[int] = None
    completion: Optional[int] = None
    total: Optional[int] = None

    def to_dict(self) -> dict[str, Optional[int]]:
        return {
            "prompt": self.prompt,
            "completion": self.completion,
            "total": self.total,
        }


def _load_cached_parse(cache_path: Path) -> tuple[dict[CommandID, ActionSequence], dict[CommandID, TokenUsage], dict[CommandID, ParseTime]] | None:
    if not cache_path.exists():
        return None

    with open(cache_path, "r", encoding="utf-8") as handle:
        payload: dict[str, Any] = json.load(handle)

    parsed_actions = {
        command_id: cast(ActionSequence, actions)
        for command_id, actions in cast(dict[str, Any], payload.get("parsed_actions", {})).items()
    }
    token_usages = {
        command_id: TokenUsage(
            prompt=cast(Optional[int], usage.get("prompt")),
            completion=cast(Optional[int], usage.get("completion")),
            total=cast(Optional[int], usage.get("total")),
        )
        for command_id, usage in cast(dict[str, Any], payload.get("token_usages", {})).items()
    }
    parse_times = {
        command_id: cast(ParseTime, value)
        for command_id, value in cast(dict[str, Any], payload.get("parse_times", {})).items()
    }

    return parsed_actions, token_usages, parse_times


def _save_cached_parse(cache_path: Path, run: int, parsed_actions: dict[CommandID, ActionSequence], token_usages: dict[CommandID, TokenUsage], parse_times: dict[CommandID, ParseTime]):
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "run": run,
        "parsed_actions": parsed_actions,
        "token_usages": {command_id: usage.to_dict() for command_id, usage in token_usages.items()},
        "parse_times": parse_times,
    }

    with open(cache_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _parse_language(user_input: str, seed: int = 1) -> tuple[ActionSequence, TokenUsage, ParseTime]:
    system_prompt = f"""
    You are a semantic parser for a cognitive robotic agent.
    You do NOT know the world map, object IDs, coordinates, or current state.

    Your task is to map the user's natural language intent to the closest matching allowed actions and extract the parameters.
    Return ONLY a JSON array containing one or more action objects and no extra text. NEVER generate conversational text, apologies, or explanations.
    Each array element must match one atomic action in order.

    IMPORTANT RULES FOR IMPLICIT TARGETS AND AMBIGUOUS COMMANDS:
    - Do NOT attempt to resolve pronouns (e.g., "it", "that") or vague nouns (e.g., "the item", "the thing") if they refer to a previously mentioned object.
    - If the user provides a vague command lacking explicit targets or destinations, STILL return a valid JSON array.
    - Simply output the detected action and omit the missing keys. 

    Allowed action values:
    - move
    - get
    - set
    - query-location
    - query-boolean
    - query-inventory

    Allowed keys (omit non-applicable keys):
    - action: string
    - desired-x: integer
    - desired-y: integer
    - direction: one of north, south, east, west
    - distance: positive integer if explicitly stated, otherwise omit
    - target-class: target class
    - target-modifiers: list of descriptive properties mentioned for target (example: ["green", "01"])
    - destination-class: destination class
    - destination-modifiers: list of descriptive properties mentioned for destination (example: ["green", "01"])

    Available classes:
    - agent
    - apple
    - cup
    - table
    - trashcan

    Inventory-query mapping rule:
    - For action query-inventory, always include target-class as agent.

    Boolean-query mapping rules:
    - Map the queried item to the target keys and the reference surface/container to the destination keys.
    
    Move vs Get vs Set distinction:
    - `move` is for PURE NAVIGATION. Use `move` ONLY when the user wants the agent to relocate and NO object manipulation is mentioned or implied.
    - `get` means PICK UP. It must NEVER have destination or destination-class keys. A `get` action MUST always specify a valid target-class. Never generate a bare `get` action.
    - `set` handles BOTH moving to a destination and placing an object you are already holding. Use `set` when the agent needs to drop an object or transport it to a location. Do NOT generate a separate `move` action before a `set`.
    - Think about physical preconditions: placing or moving an object requires first holding it. If the object to transport is explicitly mentioned, generate `get` first (with the target details), then `set` (with ONLY the destination details).
    - If the object to transport or drop is implicit (e.g., "it", "that"), assume the agent is already holding it. In this case, NEVER generate a `get` action; generate ONLY a `set` action.
    - If the user only wants to pick up, return a single `get`.
    """

    client, genai = get_client()

    config = genai.types.GenerateContentConfig(
        temperature=1.0, 
        seed=seed,
        system_instruction=system_prompt,
    )

    max_retries = 10

    llm_start = time.perf_counter()
    for attempt in range(max_retries):
        try:
            response: Any = client.models.generate_content(
                model="gemma-4-26b-a4b-it",
                config=config,
                contents=user_input
            )

            if not response:
                raise ValueError("Failed to get a response from the API.")

            response_text = response.text if response.text else ""
            response_text = response_text.replace("```json", "").replace("```", "").strip()

            try:
                parsed = json.loads(response_text)
            except json.JSONDecodeError:
                raise ValueError(f"LLM returned invalid JSON: {response_text}")

            if isinstance(parsed, dict):
                token_usage = extract_token_usage(response)
                llm_time = time.perf_counter() - llm_start
                return [cast(ActionObject, parsed)], token_usage, llm_time

            if isinstance(parsed, list):
                parsed_list = cast(list[Any], parsed)
                if all(isinstance(item, dict) for item in parsed_list):
                    token_usage = extract_token_usage(response)
                    llm_time = time.perf_counter() - llm_start
                    return cast(ActionSequence, parsed_list), token_usage, llm_time

            raise ValueError(f"LLM returned invalid format (expected object or list of objects): {parsed}")

        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(attempt)

    raise ValueError("Failed to get a valid response from the API.")


def extract_token_usage(response: Any) -> TokenUsage:
    usage: Any = getattr(response, "usage_metadata", None)

    def _read_int(field_name: str) -> Optional[int]:
        if usage is None:
            return None

        value: Any = getattr(usage, field_name, None)
        if value is None and isinstance(usage, dict):
            usage_dict = cast(dict[str, Any], usage)
            value = usage_dict.get(field_name)

        try:
            return int(cast(Union[str, int, float], value))
        except (TypeError, ValueError):
            return None

    return TokenUsage(
        prompt=_read_int("prompt_token_count"),
        completion=_read_int("candidates_token_count"),
        total=_read_int("total_token_count"),
    )


def get_client() -> tuple[Any, Any]:
    load_dotenv()
    return genai.Client(http_options={"timeout": 120 * 1000}), genai


def parse_language(run: int, dataset: list[tuple[Command, GroundTruth]], rewrite: bool = False, cache_path: Path | None = None, system_prompt: str | None = None) -> tuple[dict[CommandID, ActionSequence], dict[CommandID, TokenUsage], dict[CommandID, ParseTime]]:
    if cache_path is None:
        cache_path = Path(__file__).resolve().parent / "Tests" / "cache" / "parse_language" / f"run-{run:02d}.json"

    if not rewrite:
        cached_result = _load_cached_parse(cache_path)
        if cached_result is not None:
            return cached_result

    parsed_actions: dict[CommandID, ActionSequence] = {}
    token_usages: dict[CommandID, TokenUsage] = {}
    parse_times: dict[CommandID, ParseTime] = {}

    seed = run

    for idx, (command, _) in enumerate(dataset, 1):
        base_command_id = f"CMD-{idx:02d}"
        command_id = f"{base_command_id}-R{run:02d}"
        actions, token_usage, parse_time = _parse_language(command, seed=seed)
        parsed_actions[command_id] = actions
        token_usages[command_id] = token_usage
        parse_times[command_id] = parse_time
        print(f"\r  Parsed {idx}/{len(dataset)}", end="", flush=True)
    print()

    _save_cached_parse(cache_path, run, parsed_actions, token_usages, parse_times)

    return parsed_actions, token_usages, parse_times