"""A small AI agent that manages items by talking to Claude on Amazon Bedrock.

POST /agent  {"prompt": "Add a laptop for $999, then list everything"}

The agent loop:
  1. Send the user's prompt plus a list of tools to the model.
  2. If the model asks to use a tool, run it against DynamoDB and send back the result.
  3. Repeat until the model gives a final answer (or MAX_STEPS is reached).
"""

import json
import logging
import os
import uuid
from decimal import Decimal

import boto3

from items.app import _json_default, get_table, now_iso, response

logger = logging.getLogger()
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))

MODEL_ID = os.environ.get("MODEL_ID", "global.anthropic.claude-haiku-5-5")
MAX_STEPS = int(os.environ.get("MAX_STEPS", "6"))

SYSTEM_PROMPT = (
    "You are an inventory assistant. Use the provided tools to read and change "
    "items in the store's database. Never invent item IDs; look them up with "
    "list_items first. When you are done, reply with a short, friendly summary "
    "of what you found or changed."
)

_bedrock = None


def get_bedrock():
    global _bedrock
    if _bedrock is None:
        _bedrock = boto3.client("bedrock-runtime")
    return _bedrock


# ---------- tools the agent can use ----------

def _plain(obj):
    """DynamoDB Decimals -> plain JSON types (for sending back to the model)."""
    return json.loads(json.dumps(obj, default=_json_default))


def _to_dynamo(obj):
    """Plain JSON from the model -> DynamoDB-safe types (floats become Decimal)."""
    return json.loads(json.dumps(obj), parse_float=Decimal)


def tool_list_items(_args):
    items = get_table().scan(Limit=100).get("Items", [])
    return {"items": items, "count": len(items)}


def tool_get_item(args):
    item = get_table().get_item(Key={"id": args["id"]}).get("Item")
    return item or {"error": f"Item '{args['id']}' not found"}


def tool_create_item(args):
    timestamp = now_iso()
    item = _to_dynamo({**args.get("attributes", {}), "name": args["name"]})
    item.update({"id": str(uuid.uuid4()), "createdAt": timestamp, "updatedAt": timestamp})
    get_table().put_item(Item=item)
    return item


def tool_update_item(args):
    fields = _to_dynamo(args.get("fields", {}))
    for key in ("id", "createdAt"):
        fields.pop(key, None)
    if not fields:
        return {"error": "No fields to update"}
    if not get_table().get_item(Key={"id": args["id"]}).get("Item"):
        return {"error": f"Item '{args['id']}' not found"}
    fields["updatedAt"] = now_iso()
    names = {f"#f{i}": k for i, k in enumerate(fields)}
    values = {f":v{i}": v for i, v in enumerate(fields.values())}
    result = get_table().update_item(
        Key={"id": args["id"]},
        UpdateExpression="SET " + ", ".join(f"#f{i} = :v{i}" for i in range(len(fields))),
        ExpressionAttributeNames=names,
        ExpressionAttributeValues=values,
        ReturnValues="ALL_NEW",
    )
    return result["Attributes"]


TOOLS = {
    "list_items": (tool_list_items, "List all items in the inventory.", {
        "type": "object", "properties": {},
    }),
    "get_item": (tool_get_item, "Get one item by its id.", {
        "type": "object",
        "properties": {"id": {"type": "string", "description": "The item id"}},
        "required": ["id"],
    }),
    "create_item": (tool_create_item, "Create a new item.", {
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "Item name"},
            "attributes": {
                "type": "object",
                "description": "Other fields, e.g. {\"price\": 999, \"category\": \"electronics\"}",
            },
        },
        "required": ["name"],
    }),
    "update_item": (tool_update_item, "Update fields on an existing item.", {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "The item id"},
            "fields": {"type": "object", "description": "Fields to set, e.g. {\"price\": 899}"},
        },
        "required": ["id", "fields"],
    }),
}

TOOL_CONFIG = {
    "tools": [
        {"toolSpec": {"name": name, "description": desc, "inputSchema": {"json": schema}}}
        for name, (_fn, desc, schema) in TOOLS.items()
    ]
}


def run_tool(name, args):
    if name not in TOOLS:
        return {"error": f"Unknown tool '{name}'"}, "error"
    try:
        return _plain(TOOLS[name][0](args)), "success"
    except Exception as e:  # report tool failures to the model instead of crashing
        logger.exception("Tool %s failed", name)
        return {"error": str(e)}, "error"


# ---------- the agent loop ----------

def run_agent(prompt):
    messages = [{"role": "user", "content": [{"text": prompt}]}]
    steps = []

    for _ in range(MAX_STEPS):
        result = get_bedrock().converse(
            modelId=MODEL_ID,
            system=[{"text": SYSTEM_PROMPT}],
            messages=messages,
            toolConfig=TOOL_CONFIG,
            inferenceConfig={"maxTokens": 1024},
        )
        message = result["output"]["message"]
        messages.append(message)

        if result["stopReason"] != "tool_use":
            answer = "".join(block.get("text", "") for block in message["content"])
            return {"answer": answer.strip(), "steps": steps}

        tool_results = []
        for block in message["content"]:
            if "toolUse" not in block:
                continue
            use = block["toolUse"]
            output, status = run_tool(use["name"], use.get("input") or {})
            logger.info("Tool %s(%s) -> %s", use["name"], use.get("input"), status)
            steps.append({"tool": use["name"], "input": use.get("input"), "status": status})
            tool_results.append({
                "toolResult": {
                    "toolUseId": use["toolUseId"],
                    "content": [{"json": output}],
                    "status": status,
                }
            })
        messages.append({"role": "user", "content": tool_results})

    return {"answer": "I stopped after reaching the step limit before finishing.", "steps": steps}


def lambda_handler(event, _context):
    try:
        body = json.loads(event.get("body") or "{}")
    except json.JSONDecodeError:
        return response(400, {"error": "Request body must be valid JSON"})

    prompt = body.get("prompt") if isinstance(body, dict) else None
    if not prompt or not isinstance(prompt, str):
        return response(400, {"error": "'prompt' is required and must be a string"})
    if len(prompt) > 2000:
        return response(400, {"error": "'prompt' must be 2000 characters or fewer"})

    try:
        return response(200, run_agent(prompt))
    except Exception:
        logger.exception("Agent failed")
        return response(500, {"error": "The agent could not complete the request"})