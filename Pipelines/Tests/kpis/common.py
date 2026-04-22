from typing import Any, Optional, Union, cast


def _normalize_value(val: Any) -> Optional[Union[str, list[str]]]:
    if val is None:
        return None

    if isinstance(val, list):
        val_as_list = cast(list[Any], val)
        return sorted([str(v).lower().strip() for v in val_as_list])

    return str(val).lower().strip()


def _normalize_action(action: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in (action or {}).items():
        normalized[key] = _normalize_value(value)
    return normalized


def _compare_keys(parsed: dict[str, Any], gold: dict[str, Any], key: str) -> bool:
    p_val = (parsed or {}).get(key)
    g_val = (gold or {}).get(key)

    if p_val is None and g_val is None:
        return True
    if (p_val is None) != (g_val is None):
        return False

    return _normalize_value(p_val) == _normalize_value(g_val)


def exact_match(parsed: list[dict[str, Any]], gold_standard: list[dict[str, Any]]) -> bool:
    if len(parsed or []) != len(gold_standard or []):
        return False

    for p, g in zip(parsed or [], gold_standard or []):
        if _normalize_action(p) != _normalize_action(g):
            return False

    return True


def partial_correctness(parsed: list[dict[str, Any]], gold_standard: list[dict[str, Any]]) -> dict[str, float]:
    scores = {"action": 0.0, "object_identification": 0.0, "relation": 0.0}
    max_len = max(len(parsed or []), len(gold_standard or []))

    action_correct = 0
    object_correct = 0
    relation_correct = 0

    for i in range(max_len):
        p = parsed[i] if i < len(parsed or []) else {}
        g = gold_standard[i] if i < len(gold_standard or []) else {}

        if p.get("action") == g.get("action"):
            action_correct += 1

        if (_compare_keys(p, g, "target-class") and _compare_keys(p, g, "target-modifiers")):
            object_correct += 1

        if (_compare_keys(p, g, "desired-x") and _compare_keys(p, g, "desired-y") and
            _compare_keys(p, g, "destination") and _compare_keys(p, g, "destination-class") and
            _compare_keys(p, g, "destination-modifiers")):
            relation_correct += 1

    if max_len > 0:
        scores["action"] = action_correct / max_len
        scores["object_identification"] = object_correct / max_len
        scores["relation"] = relation_correct / max_len

    return scores
