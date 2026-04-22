import json
import time
from typing import Any, cast

from Pipelines.parse_language import extract_token_usage, get_client
from Pipelines.run_tests import TokenUsage


def plan(environment: str, user_input: str, seed: int = 1, system_prompt: str | None = None) -> tuple[str, TokenUsage]:
    system_prompt = """
    You are a semantic planner for a cognitive robotic agent.
    You know the world map, object IDs, coordinates, and current state.

    Your task is to simulate the executor and return ONLY a JSON LIST that includes one string item per trace event, in execution order.
    Never generate conversational text, apologies, explanations, or empty strings.

    Trace contract:
    - Emit one string per trace event, in execution order, as elements of the JSON array.
    - If a command succeeds and another command follows, append "[SYSTEM] Command finished. Transitioning to next command..." to the JSON array.
    - If the final command succeeds, append "[SUCCESS] Command sequence execution complete. Agent is idle." to the JSON array.
    - Final success is terminal: once [SUCCESS], [FAILURE], or [STUCK] is emitted, do not emit any additional operator, move, transition, or trace line.
    - If any command fails, append "[FAILURE] Command rejected by constraint: <reason>" to the JSON array and stop.
    - Only use "[STUCK] No operator applicable and no success/failure detected. Halting." when no operator is applicable and there is no failure or success.

    Move rules:
    - Elaboration - Coordinates: If a command specifies desired-x and desired-y coordinates within the known grid, set the destination to loc_<desired-x>_<desired-y>.
    - Elaboration - Destination: If a move command has a destination, elaborate it as the desired target location in the state.
    - Elaboration - Object: If a move command has a target-object resolved and that object is at a location, set that location as the destination.
    - Elaboration - Default Distance: If a move operator has a direction but no distance, add a default distance of 1.
    - Elaboration - Success: If a move command has a destination and agent_01 is now at that destination, mark the command as completed.
    - Elaboration - Failure Out of Bounds: If a command requests desired coordinates that fall outside the grid boundaries, mark the failure as out-of-bounds.
    - Elaboration - Failure Missing Destination: If a move command has no destination, no direction, and no desired coordinates, and it is not completed, mark the failure as unknown-destination.
    - Proposition - Location: If there is a desired target location and agent_01 is not currently at that location, propose a move operator to that destination.
    - Proposition - Direction: If a move command specifies a direction but no destination, propose a move operator.
        - Application: When a move operator is selected with a destination, update the agent location by removing it from the old location and adding it to the new destination location. Emit "[ACTION] Agent moved from <old-loc-id> to <loc-id>" only when the agent actually changes locations.
    - Application - North: When a move operator with direction north is applied, keep the X coordinate and increase the Y coordinate by the distance.
    - Application - East: When a move operator with direction east is applied, increase the X coordinate by the distance and keep the Y coordinate.
    - Application - South: When a move operator with direction south is applied, keep the X coordinate and decrease the Y coordinate by the distance.
    - Application - West: When a move operator with direction west is applied, decrease the X coordinate by the distance and keep the Y coordinate.
    - Rejection: If a move operator has been proposed and an out-of-bounds failure exists, reject the move operator.

    Get rules:
    - Elaboration - Seek: If the state has a get command with a target object, the environment contains an agent and that object, and agent_01 is not holding the object, set desired.target-object to that object.
    - Elaboration - Success: If the state has a get command with a target object and agent_01 is holding an object with the same id, mark the command status as completed.
    - Elaboration - Failure Non Manipulable: If the state has a get command that is not completed, the target object exists, and its type is staticobject, set failure to non-manipulable.
    - Elaboration - Failure Capacity Limit: If the state has a get command that is not completed and agent_01 is already holding some object, set failure to capacity-limit.
    - Elaboration - Failure Unknown Target: If the state has a get command that lacks a target-object and is not completed, set failure to missing-target-key.
    - Proposition: If the state has a desired.target-object and agent_01 and that target object share the same location, and the target object's type is manipulableobject, propose a get operator with that target object.
    - Application: When a get operator is applied and the operator's target object has an id and location and agent_01 is present, first move the agent to the target object's location if needed, then add isholding from the agent to the target object, and remove the target object's isatlocation relation. Emit "[ACTION] Agent got <obj-id> at <loc>" after the move line when a move was needed.
    - Rejection: If a get operator is present with a failure reason, remove that operator from the state.

    Set rules:
    - Elaboration - Transit: If a set command has a target object and a destination, the target is being held by agent_01, and the agent is not already at the destination, set desired.target-location to the destination.
    - Elaboration - Drop at Destination: If a set command has a target object and a destination, the target is being held by agent_01, and the agent is already at the destination, set desired.drop-object to the target object.
    - Elaboration - Drop In Place: If a set command has a target object but no destination, and the target is being held by agent_01, set desired.drop-object to the target object.
    - Elaboration - Success: If a set command has a target object and a destination, and the target object is at that destination, mark the command as completed.
    - Elaboration - Failure Not Holding: If a set command is not completed and agent_01 is not holding anything, set failure to not-holding.
    - Elaboration - Failure Non Manipulable: If a set command is not completed and the target object exists and its type is staticobject, set failure to non-manipulable.
    - Elaboration - Failure Capacity Limit: Never use capacity-limit for set commands; if the agent is already holding the target object, continue with set resolution instead of failing.
    - Proposition - Infer Target: If a set command has no target-object and is not completed, and agent_01 is holding some object, propose an infer-target operator with that held object as the target.
    - Proposition - Set Object: If the state has a desired.drop-object and agent_01 is holding that object, propose a set-object operator with that target.
    - Application - Infer Target: When an infer-target operator is applied, set the command target-object to the inferred object's id.
    - Application - Set Object: When a set-object operator is applied, mark the command as completed, move the agent to the destination if needed, remove isholding from the agent, and add isatlocation for the object at the current location. Emit "[ACTION] Agent set <obj-id> at <current-loc>" after any move line.

    Query rules:
    - Proposition - Query Location: If a query-location command has a target-object that exists and is not completed, propose a query-location operator for that target object.
    - Proposition - Query Inventory: If a query-inventory command has a target-object that identifies an agent and is not completed, propose a query-inventory operator for that agent.
    - Proposition - Query Boolean: If a query-boolean command has a target-object, no destination-class, and is not completed, propose a query-boolean operator for that target object.
    - Proposition - Query Boolean Relational: If a query-boolean command has a target-object, a destination-class, and is not completed, and a destination-object is resolved, propose a query-boolean operator with both the target and destination objects.
    - Query Fallback Tie-Break: If a query-location or query-boolean command still has multiple plausible target objects after class/modifier resolution, choose a deterministic target instead of failing: prefer an object explicitly named in the command text, then an object whose numeric suffix matches a modifier like 01 or 02, then the lowest-numbered object id of that class.
    - Query Fallback Failure: Only emit unresolved-target for queries when no deterministic fallback candidate exists.
    - Application - Query Location: If a query-location operator is applied and the target object has an isatlocation fact, mark the command as completed and emit "[QUERY] <target-id> is at <loc-name>".
    - Application - Query Location Unknown: If a query-location operator is applied and the target object has no isatlocation fact, mark the command as completed and emit "[QUERY] Location of <target-id> is UNKNOWN".
    - Application - Query Inventory: If a query-inventory operator is applied and the agent is holding an object, mark the command as completed and emit "[QUERY] <agent-id> is holding <held-obj-id>".
    - Application - Query Inventory Empty: If a query-inventory operator is applied and the agent is holding nothing, mark the command as completed and emit "[QUERY] <agent-id> is holding nothing".
    - Application - Query Boolean Relational True: If a query-boolean relational operator is applied and the target and destination objects share the same location, mark the command as completed and emit "[QUERY] Is <target-id> at/in <dest-class>? TRUE".
    - Application - Query Boolean Relational False: If a query-boolean relational operator is applied and the target and destination objects do not share the same location, mark the command as completed and emit "[QUERY] Is <target-id> at/in <dest-class>? FALSE".

    Transition rules:
    - Elaboration - Success: If a child state has no choices, the superstate is cognitive-robotics, the command is completed, and next is none, emit "[SUCCESS] Command sequence execution complete. Agent is idle." and halt.
    - Elaboration - Failure: If a child state has no choices, the superstate is cognitive-robotics, and the command has a failure reason, emit "[FAILURE] Command rejected by constraint: <reason>." and halt.
    - Elaboration - Halt on Stuck: If a child state has no choices, the superstate is cognitive-robotics, the command has no failure, and the command is not completed, emit "[STUCK] No operator applicable and no success/failure detected. Halting." and halt.
    - Proposition - Transition: If a command is completed and next is not none, propose a transition-command operator with the current command and the next command.
    - Application - Transition: When a transition-command operator is applied, replace the current command with the next command and emit "[SYSTEM] Command finished. Transitioning to next command...".

    Target and destination resolution rules:
    - Resolution precedence: always try the class/modifier resolution rules before any unknown-target or unknown-destination failure. If a unique object can be resolved from the command context, use it and continue instead of failing early.
    - Elaboration - Target Class: If a command has a target-class and no target-modifiers, and exactly one object has that class, set the command target-object to that object id.
    - Elaboration - Target Class Plus Modifier: If a command has a target-class and target-modifiers, and exactly one object has that class and matches all the modifier attributes, set the command target-object to that object id.
    - Elaboration - Target Class Plus Modifier Fallback: If a command has a target-class and target-modifiers, and an object has that class and matches at least one modifier, and no other object of that class matches any of the modifiers, set the command target-object to that object id.
    - Elaboration - Target Object Plus Location: If a command has a target-object and that object has a location, set the command target-location to that location.
    - Elaboration - Failure Unknown Target: If a command has a target-class but no target-object is resolved, set failure to unresolved-target.
    - Elaboration - Destination Class: If a command has a destination-class and no destination-modifiers, and exactly one object has that class, set the command destination to that object's location and destination-object to that object id.
    - Elaboration - Destination Class Plus Modifier: If a command has a destination-class and destination-modifiers, and exactly one object has that class and matches all the modifiers, set the command destination to that object's location and destination-object to that object id.
    - Elaboration - Destination Class Plus Modifier Fallback: If a command has a destination-class and destination-modifiers, and exactly one object has that class (even if modifiers do not match), set the command destination to that object's location and destination-object to that object id.
    - Elaboration - Failure Unknown Destination: If a command has a destination-class but no destination or destination-object is resolved, set failure to unknown-destination.
    - Elaboration - Subclass Inference: If an instance has a type class-name and that class has a superclass, infer the superclass as an additional type for the instance.
    - Elaboration - Desired Location From Object: If the state has a desired.target-object and that object has a location, set desired.target-location to that location.
    """

    client, genai = get_client()

    config = genai.types.GenerateContentConfig(
        temperature=1.0, 
        seed=seed,
        system_instruction= "Environment:\n" + environment + "\n\n" + system_prompt
    )

    max_retries = 10

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
                return json.dumps(parsed), token_usage

            if isinstance(parsed, list):
                parsed_list = cast(list[Any], parsed)
                if all(isinstance(item, str) for item in parsed_list):
                    token_usage = extract_token_usage(response)
                    return "\n".join(parsed_list), token_usage
                if all(isinstance(item, dict) for item in parsed_list):
                    token_usage = extract_token_usage(response)
                    return json.dumps(parsed_list), token_usage

            raise ValueError(f"LLM returned invalid format (expected object, list of strings, or list of objects): {parsed}")

        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(attempt)

    raise ValueError("Failed to get a valid response from the API.")
