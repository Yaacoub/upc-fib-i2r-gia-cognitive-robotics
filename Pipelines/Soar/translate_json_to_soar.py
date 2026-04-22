from typing import cast

from Pipelines.parse_language import ActionSequence, CommandID
from Pipelines.run_tests import Script


def _build_run_script(actions: ActionSequence) -> str:
    lines = [
        "source rules/cognitive-robotics.soar",
        "source rules/elaborations.soar",
        "source rules/move.soar",
        "source rules/get.soar",
        "source rules/set.soar",
        "source rules/transition.soar",
        "source rules/query.soar",
    ]

    lines.append(_translate_json_to_soar(actions))
    lines.append("run")

    return "\n".join(lines) + "\n"


def _translate_json_to_soar(actions: ActionSequence) -> str:
    rule_lines = [
        "sp {apply*init-environment*commands",
        "    (state <s> ^operator <o>)",
        "    (<o> ^name init-environment)",
        "-->",
        "    (<s> ^command <cmd1>)"
    ]

    for i, action in enumerate(actions, start=1):
        cmd_id = f"<cmd{i}>"
        command = dict(action)

        x_val = command.get("desired-x")
        y_val = command.get("desired-y")
        if command.get("destination") is None and x_val is not None and y_val is not None:
            command["destination"] = f"loc_{int(x_val)}_{int(y_val)}"

        for key, value in command.items():
            if isinstance(value, list):
                for item in cast(list[str], value):
                    if str(item).strip() != "":
                        item_str = str(item).strip().lower().replace(" ", "_")
                        rule_lines.append(f"    ({cmd_id} ^{key} {item_str})")
            elif value is not None and str(value).strip() != "":
                value_str = str(value).strip().lower().replace(" ", "_")
                rule_lines.append(f"    ({cmd_id} ^{key} {value_str})")

        if i < len(actions):
            rule_lines.append(f"    ({cmd_id} ^next <cmd{i+1}>)")
        else:
            rule_lines.append(f"    ({cmd_id} ^next none)")

    rule_lines.append("}")
    return "\n".join(rule_lines)


def translate_json_to_soar(actions: dict[CommandID, ActionSequence]) -> dict[CommandID, Script]:
    target_scripts: dict[CommandID, Script] = {}

    for command_id, action in actions.items():
        script = _build_run_script(action)
        target_scripts[command_id] = script

    return target_scripts