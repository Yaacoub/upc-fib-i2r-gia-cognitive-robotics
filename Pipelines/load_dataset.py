import json
from pathlib import Path
from typing import Any, Optional


Command = str
GroundTruth = list[dict[str, Any]]


def load_dataset(max: Optional[int] = None) -> list[tuple[Command, GroundTruth]]:
        commands_path = Path(__file__).resolve().parent.parent / "Dataset Creation" / "commands.txt"
        ground_truth_path = Path(__file__).resolve().parent.parent / "Dataset Creation" / "ground_truth.txt"

        with open(commands_path, "r") as f:
            commands = [line.strip() for line in f if line.strip() and not line.strip().startswith("#")]

        with open(ground_truth_path, "r") as f:
            ground_truths = [json.loads(line.strip()) for line in f if line.strip()]

        commands = commands[:max] if max is not None else commands
        ground_truths = ground_truths[:max] if max is not None else ground_truths

        return list(zip(commands, ground_truths))