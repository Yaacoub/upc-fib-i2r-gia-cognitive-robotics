import json
import os
import subprocess
import time

from dotenv import load_dotenv
from google import genai
from typing import Any, cast, Optional


ActionObject = dict[str, Any]
ActionSequence = list[ActionObject]
TokenUsage = dict[str, Optional[int]]



# To get GEMINI_API_KEY: https://aistudio.google.com/app/api-keys
load_dotenv()
client = genai.Client()



def _normalize_symbol(value: str) -> str:
    return value.strip().lower().replace(" ", "_")



def append_to_log(log_path: str, text: str):
    with open(log_path, "a") as f:
        f.write(text)
        if not text.endswith("\n"):
            f.write("\n")


def _extract_token_usage(response: Any) -> TokenUsage:
    usage = getattr(response, "usage_metadata", None)

    def _read_int(field_name: str) -> Optional[int]:
        if usage is None:
            return None

        value = getattr(usage, field_name, None)
        if value is None and isinstance(usage, dict):
            value = usage.get(field_name)

        try:
            return int(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    return {
        "prompt": _read_int("prompt_token_count"),
        "completion": _read_int("candidates_token_count"),
        "total": _read_int("total_token_count"),
    }


def _format_token_usage(token_usage: TokenUsage) -> str:
    prompt = token_usage.get("prompt")
    completion = token_usage.get("completion")
    total = token_usage.get("total")

    prompt_str = str(prompt) if prompt is not None else "n/a"
    completion_str = str(completion) if completion is not None else "n/a"
    total_str = str(total) if total is not None else "n/a"

    return f"input={prompt_str}, output={completion_str}, total={total_str}"



def parse_natural_language(user_input: str) -> tuple[ActionSequence, TokenUsage]:
    prompt = f"""
    You are a semantic parser for a cognitive robotic agent.
    You do NOT know the world map, object IDs, coordinates, or current state.

    Your task is to map the user's natural language intent to the closest matching allowed actions and extract the parameters.
    Return ONLY a JSON array containing one or more action objects and no extra text.
    Each array element must match one atomic action in order.

    IMPORTANT RULES FOR IMPLICIT TARGETS:
    - Do NOT attempt to resolve pronouns (e.g., "it", "that") or vague nouns (e.g., "the item", "the thing"). 
    - If the user says "drop it" or "set the item down", completely OMIT the target-class and target-modifiers keys from the JSON. The agent's physical engine will infer the target based on what it is currently holding.

    Allowed action values:
    - move
    - get (Pick up / grasp an object. The goal is HOLDING it. Never include destination keys.)
    - set (Place / deliver / drop / put down an object. If a destination is specified, include destination keys. If dropping in place, omit destination keys.)
    - query-location
    - query-boolean (Use this for yes/no questions about spatial relations or containment)
    - query-inventory

    Allowed keys (omit non-applicable keys):
    - action: string
    - desired-x: integer (if coordinates are provided)
    - desired-y: integer (if coordinates are provided)
    - direction: one of north, south, east, west
    - distance: positive integer if explicitly stated, otherwise omit
    - target-class: target class
    - target-modifiers: list of descriptive properties mentioned for target (example: ["green"])
    - destination-class: destination class
    - destination-modifiers: list of descriptive properties mentioned for destination (example: ["green"])

    Available classes:
    - agent (the robot/agent, "you", "yourself", etc.)
    - apple (fruit objects)
    - cup (container objects)
    - table (static surface)
    - trashcan (static container)
    - gridlocation (map location)

    Inventory-query mapping rule:
    - For action query-inventory, always include target-class as agent.

    Boolean-query mapping rules:
    - Map the queried item to the target keys and the reference surface/container to the destination keys.
    
    Get vs Set distinction (CRITICAL):
    - `get` means PICK UP. It must NEVER have destination or destination-class keys.
    - `set` ONLY handles placing an object you are already holding.
    - If a command implies picking up an object and moving it somewhere (e.g., "throw the cup in the trash", "put the cup on the table", "bring the apple to me"), you MUST return TWO separate actions: first a `get`, then a `set`.
    - If the user only wants to pick up (e.g. "pick up the apple", "grab the cup", "grasp the red apple"), return a single `get`.

    Command: "{user_input}"
    """

    # for model in client.models.list():
    #     print(f"Available model: {model.name}")

    response = client.models.generate_content(
        model="gemma-4-26b-a4b-it",
        contents=prompt
    )

    token_usage = _extract_token_usage(response)

    response_text = response.text if response.text else ""
    response_text = response_text.replace("```json", "").replace("```", "").strip()
    parsed = json.loads(response_text)

    if isinstance(parsed, dict):
        return [cast(ActionObject, parsed)], token_usage

    if isinstance(parsed, list):
        parsed_list = cast(list[Any], parsed)
        if parsed_list and all(isinstance(item, dict) for item in parsed_list):
            return cast(ActionSequence, parsed_list), token_usage

    raise ValueError(f"LLM returned invalid format (expected object or non-empty array): {parsed}")



def build_soar_command_rules(actions: ActionSequence) -> str:
    rule_lines = [
        "sp {apply*init-environment*commands",
        "    (state <s> ^operator <o>)",
        "    (<o> ^name init-environment)",
        "-->",
        "    (<s> ^command <cmd1>)"
    ]

    for i, action in enumerate(actions, start=1):
        cmd_id = f"<cmd{i}>"

        # Preserve explicit coordinate intents
        x_val = action.get("desired-x")
        y_val = action.get("desired-y")
        if action.get("destination") is None and x_val is not None and y_val is not None:
            action["destination"] = f"loc_{int(x_val)}_{int(y_val)}"

        for key, value in action.items():
            if isinstance(value, list):
                for item in cast(list[str], value):
                    if item is not None and str(item).strip() != "":
                        rule_lines.append(f"    ({cmd_id} ^{key} {_normalize_symbol(str(item))})")
            elif value is not None and str(value).strip() != "":
                rule_lines.append(f"    ({cmd_id} ^{key} {_normalize_symbol(str(value))})")

        # Create a linked list pointing to the next command
        if i < len(actions):
            rule_lines.append(f"    ({cmd_id} ^next <cmd{i+1}>)")
        else:
            rule_lines.append(f"    ({cmd_id} ^next none)")

    rule_lines.append("}")
    return "\n".join(rule_lines)



def build_run_script(actions: ActionSequence) -> str:
    lines = [
        "source cognitive-robotics.soar",
        "source elaborations.soar",
        "source move.soar",
        "source get.soar",
        "source set.soar",
        "source transition.soar",
        "source query.soar",
    ]

    lines.append(build_soar_command_rules(actions))
    lines.append("run")

    return "\n".join(lines) + "\n"



def filter_soar_output(raw_output: str) -> str:
    lines = raw_output.split("\n")
    filtered_lines: list[str] = []
    
    skip_keywords = [
        "Soar Command Line Interface", "Launching the Soar Cognitive Architecture",
        "...created Soar kernel", "...created agent", "Soar CLI in single agent mode",
        "soar %", "System halted", "Interrupt received", "This Agent halted", "****",
        "Trace level 1 enabled", "For a full list of trace options", "An agent halted during the run",
        "Total: ", "productions sourced", "production excised", "-->", "Sourcing ",
        "--- STEP-", "--- CMD-", "Run stopped"
    ]
    
    for line in lines:
        stripped = line.strip()
        if not stripped: continue
        if stripped == "|" or stripped == "*": continue
        if any(kw in line for kw in skip_keywords): continue
        if line.startswith("     ") and ":" in line and "O:" not in line and "==>S:" not in line: continue
        
        filtered_lines.append(line)
        
    return "\n".join(filtered_lines)



if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    log_path = os.path.join(base_dir, "analysis_output.log")
    commands_path = os.path.join(base_dir, "..", "Dataset Creation", "commands.txt")
    run_path = os.path.join(base_dir, "run.soar")
    soar_cli_path = os.path.join(base_dir, "SoarSuite_9.6.4-Multiplatform", "SoarCLI.sh")

    with open(log_path, "w") as f:
        f.write("")

    with open(commands_path, "r") as f:
        commands = [line.strip() for line in f.readlines() if line.strip() and not line.strip().startswith("#")]

    total_tokens: TokenUsage = {"prompt": 0, "completion": 0, "total": 0}

    start_time = time.perf_counter()
    
    for index, command in enumerate(commands):
        print(f"Executing CMD-{index+1}/{len(commands)}: {command}")
        
        append_to_log(log_path, "="*50)
        append_to_log(log_path, f"CMD-{index+1}: {command}")
        append_to_log(log_path, "="*50)

        try:
            parsed_actions, token_usage = parse_natural_language(command)
            append_to_log(log_path, f"LLM Output ({len(parsed_actions)} step(s)):\n{json.dumps(parsed_actions, indent=2)}\n\n")
            append_to_log(log_path, f"LLM Token Usage: {_format_token_usage(token_usage)}\n\n")

            for key in ("prompt", "completion", "total"):
                token_value = token_usage.get(key)
                if token_value is not None:
                    current_value = total_tokens.get(key)
                    total_tokens[key] = (current_value or 0) + token_value

            run_script = build_run_script(parsed_actions)
            with open(run_path, "w") as f:
                f.write(run_script)

            soar_rules_output = build_soar_command_rules(parsed_actions) + "\n"
            append_to_log(log_path, f"Soar Output (Soar Code):\n{soar_rules_output}\n")

            process = subprocess.run(
                [soar_cli_path],
                input="source run.soar\n",
                capture_output=True,
                text=True,
                cwd=base_dir,
                timeout=60
            )
            
            soar_output = filter_soar_output(process.stdout)
            if not soar_output.strip():
                soar_output = "[No Soar execution output. Check if it hit a constraint silently or timed out.]"
            
            append_to_log(log_path, f"Soar return code: {process.returncode}")
            if process.stderr and process.stderr.strip():
                append_to_log(log_path, f"Soar stderr:\n{process.stderr}\n")
            append_to_log(log_path, f"Soar Execution:\n{soar_output}\n\n")
            
        except subprocess.TimeoutExpired:
            append_to_log(log_path, "Error: Soar CLI timed out after 60 seconds\n")
            print("  -> Error: Soar CLI timed out after 60 seconds")
        except Exception as e:
            append_to_log(log_path, f"Error: {e}\n\n")
            print(f"  -> Error: {e}")

        finally:
            if os.path.exists(run_path):
                os.remove(run_path)

    elapsed_seconds = time.perf_counter() - start_time
    append_to_log(log_path, f"Overall elapsed time: {elapsed_seconds:.2f} seconds")
    append_to_log(log_path, f"LLM Token Totals: {_format_token_usage(total_tokens)}")