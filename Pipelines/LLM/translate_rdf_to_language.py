from pathlib import Path

from Pipelines.parse_ontology import parse_ontology


def translate_rdf_to_language(rewrite: bool = False) -> None:
    txt_file_path = Path(__file__).resolve().parent / "cognitive-robotics.txt"
    if txt_file_path.exists() and not rewrite:
        return

    txt_file_path.parent.mkdir(parents=True, exist_ok=True)

    entities = parse_ontology()
    lines: list[str] = []
    lines.append("## Environment (OWL-derived initial state)")
    lines.append("Grid is 4x4 with coordinates [0,0] bottom-left to [3,3] top-right.")

    agent_loc = entities.get("agent_01", {}).get("isatlocation", [None])[0]
    held = entities.get("agent_01", {}).get("isholding", [None])[0]

    if agent_loc is not None:
        x = entities.get(agent_loc, {}).get("hasxcoordinate", ["?"])[0]
        y = entities.get(agent_loc, {}).get("hasycoordinate", ["?"])[0]
        coord = f"[{x},{y}]"
        lines.append(f"- Agent is at location {agent_loc} coordinates {coord}.")
    else:
        lines.append("- Agent location: unknown.")

    if held is not None:
        lines.append(f"- Agent is holding: {held}.")
    else:
        lines.append("- Agent is holding: nothing.")

    lines.append("- Objects (excluding pure grid cells):")
    for ind, attrs in entities.items():
        types = attrs.get("type", [])
        if "gridlocation" in types:
            continue
        if any(t in types for t in ["agent", "class", "datatypeproperty", "objectproperty", "functionalproperty", "ontology"]):
            continue
        if ind == "cognitive-robotics" or not attrs:
            continue
        
        loc = attrs.get("isatlocation", [None])[0]
        x, y = None, None
        if loc:
            loc_attrs = entities.get(loc, {})
            x_list = loc_attrs.get("hasxcoordinate")
            y_list = loc_attrs.get("hasycoordinate")
            if x_list and y_list:
                x = x_list[0]
                y = y_list[0]
        
        coord_str = f" at [{x},{y}]" if x is not None and y is not None else ""
        loc_str = f" on {loc}" if loc else " (location unknown)"
        
        color = attrs.get("hascolor", [None])[0]
        valid_types = sorted([t for t in attrs.get("type", []) if t != "namedindividual"])
        type_hint = ", ".join(valid_types)
        color_str = f", color={color}" if color else ""
        
        lines.append(f"  - {ind}: types [{type_hint}]{color_str}{loc_str}{coord_str}.")

    text_content = "\n".join(lines) + "\n"

    with open(txt_file_path, "w", encoding="utf-8") as f:
        f.write(text_content)