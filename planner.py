
import requests
import json

MODEL = "qwen3:4b"
OLLAMA_URL = "http://localhost:11434/api/chat"


# Enforce the expected structure of the plan
PLAN_SCHEMA = {
    "type": "object",
    "properties": {
        "goal": {
            "type": "string"
        },
        "tasks": {
            "type": "array",
            "minItems": 3,
            "maxItems": 7,
            "items": {
                "type": "object",
                "properties": {
                    "id": {
                        "type": "integer"
                    },
                    "description": {
                        "type": "string"
                    },
                    "status": {
                        "type": "string",
                        "enum": ["pending"]
                    }
                },
                "required": [
                    "id",
                    "description",
                    "status"
                ],
                "additionalProperties": False
            }
        }
    },
    "required": [
        "goal",
        "tasks"
    ],
    "additionalProperties": False
}


def create_plan(mission: str) -> dict:

    system_prompt = """
    You are WebPilot's research planning agent.

    Convert the user's research mission into 3 to 7 actionable tasks.

    Rules:
    - Break the mission into logically ordered steps.
    - Start with information discovery.
    - Include gathering information from relevant sources.
    - Include verification of findings.
    - Include compiling and saving the report.
    - Each task must be specific and actionable.
    - Do not execute the tasks.
    """

    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": mission
            }
        ],
        "stream": False,
        "think": False,
        "format": PLAN_SCHEMA,
        "options": {
            "num_predict": 900,
            "temperature": 0
        }
    }

    try:
        print("\nConnecting to Ollama...")
        print("Generating plan using Qwen...")

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=(10, 120)
        )

        response.raise_for_status()

        result = response.json()

        # Display generation metadata
        print("\n========== OLLAMA METADATA ==========")
        print("Done:", result.get("done"))
        print("Done reason:", result.get("done_reason"))
        print("Output tokens:", result.get("eval_count"))
        print("=====================================")

        raw_content = result["message"]["content"]

        if not raw_content.strip():
            print("\nQwen returned an empty response.")
            print(json.dumps(result, indent=2))
            return {}

        print("\n========== RAW QWEN RESPONSE ==========")
        print(raw_content)
        print("=======================================")

        # Parse JSON
        plan = json.loads(raw_content)

        # Validate plan
        if not isinstance(plan, dict):
            raise ValueError("Plan must be a JSON object")

        if not isinstance(plan.get("goal"), str):
            raise ValueError("Goal must be a string")

        if not isinstance(plan.get("tasks"), list):
            raise ValueError("Tasks must be a list")

        if not 3 <= len(plan["tasks"]) <= 7:
            raise ValueError("Plan must contain 3 to 7 tasks")

        for index, task in enumerate(plan["tasks"], start=1):

            if not isinstance(task, dict):
                raise ValueError(
                    f"Task {index} must be an object"
                )

            required_fields = [
                "id",
                "description",
                "status"
            ]

            for field in required_fields:
                if field not in task:
                    raise ValueError(
                        f"Task {index} is missing '{field}'"
                    )

            if not isinstance(task["id"], int):
                raise ValueError(
                    f"Task {index}: id must be an integer"
                )

            if not isinstance(task["description"], str):
                raise ValueError(
                    f"Task {index}: description must be a string"
                )

            if task["status"] != "pending":
                raise ValueError(
                    f"Task {index}: status must be pending"
                )

        print("\nPlan generated successfully!")

        return plan

    except requests.exceptions.ReadTimeout:
        print("Error: Ollama took longer than 5 minutes.")

    except requests.exceptions.ConnectionError:
        print("Error: Ollama is not running.")

    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        print(response.text)

    except json.JSONDecodeError as e:
        print(f"JSON Parsing Error: {e}")

    except ValueError as e:
        print(f"Plan Validation Error: {e}")

    except KeyError as e:
        print(f"Unexpected Ollama response structure: {e}")

    return {}


def save_plan(plan: dict, filename: str = "generated_plan.json"):

    with open(
        filename,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            plan,
            file,
            indent=2,
            ensure_ascii=False
        )

    print(f"\nPlan saved to {filename}")


if __name__ == "__main__":

    mission = input(
        "\nEnter your research mission: "
    )

    plan = create_plan(mission)

    if plan:

        print("\n========== GENERATED PLAN ==========\n")

        print(
            json.dumps(
                plan,
                indent=2,
                ensure_ascii=False
            )
        )

        save_plan(plan)
