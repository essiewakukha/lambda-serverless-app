"""Lambda handler for a simple Items CRUD API (API Gateway REST proxy events)."""

import json
import logging
import os
import uuid
from datetime import datetime, timezone
from decimal import Decimal

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger()
logger.setLevel(os.environ.get("LOG_LEVEL", "INFO"))

_table = None


def get_table():
    """Create the DynamoDB table client lazily (reused across warm invocations)."""
    global _table
    if _table is None:
        _table = boto3.resource("dynamodb").Table(os.environ["TABLE_NAME"])
    return _table


def _json_default(obj):
    if isinstance(obj, Decimal):
        return int(obj) if obj % 1 == 0 else float(obj)
    raise TypeError(f"Not serializable: {type(obj)}")


def response(status, body=None):
    return {
        "statusCode": status,
        "headers": {
            "Content-Type": "application/json",
            "Access-Control-Allow-Origin": "*",
        },
        "body": json.dumps(body, default=_json_default) if body is not None else "",
    }


def parse_body(event):
    raw = event.get("body") or "{}"
    try:
        data = json.loads(raw, parse_float=Decimal)
    except json.JSONDecodeError:
        raise ValueError("Request body must be valid JSON")
    if not isinstance(data, dict):
        raise ValueError("Request body must be a JSON object")
    return data


def now_iso():
    return datetime.now(timezone.utc).isoformat()


# ---------- route handlers ----------

def health(_event):
    return response(200, {"status": "ok", "stage": os.environ.get("STAGE", "local")})


def list_items(_event):
    result = get_table().scan(Limit=100)
    return response(200, {"items": result.get("Items", []), "count": result.get("Count", 0)})


def create_item(event):
    data = parse_body(event)
    name = data.get("name")
    if not name or not isinstance(name, str):
        return response(400, {"error": "'name' is required and must be a string"})

    timestamp = now_iso()
    item = {
        **data,
        "id": str(uuid.uuid4()),
        "name": name,
        "createdAt": timestamp,
        "updatedAt": timestamp,
    }
    get_table().put_item(Item=item)
    return response(201, item)


def get_item(event):
    item_id = event["pathParameters"]["id"]
    item = get_table().get_item(Key={"id": item_id}).get("Item")
    if not item:
        return response(404, {"error": f"Item '{item_id}' not found"})
    return response(200, item)


def update_item(event):
    item_id = event["pathParameters"]["id"]
    data = parse_body(event)
    data.pop("id", None)
    data.pop("createdAt", None)
    if not data:
        return response(400, {"error": "Provide at least one field to update"})

    data["updatedAt"] = now_iso()
    names = {f"#f{i}": k for i, k in enumerate(data)}
    values = {f":v{i}": v for i, v in enumerate(data.values())}
    expr = "SET " + ", ".join(f"#f{i} = :v{i}" for i in range(len(data)))

    try:
        result = get_table().update_item(
            Key={"id": item_id},
            UpdateExpression=expr,
            ExpressionAttributeNames=names,
            ExpressionAttributeValues=values,
            ConditionExpression="attribute_exists(id)",
            ReturnValues="ALL_NEW",
        )
    except ClientError as e:
        if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
            return response(404, {"error": f"Item '{item_id}' not found"})
        raise
    return response(200, result["Attributes"])


def delete_item(event):
    item_id = event["pathParameters"]["id"]
    try:
        get_table().delete_item(
            Key={"id": item_id}, ConditionExpression="attribute_exists(id)"
        )
    except ClientError as e:
        if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
            return response(404, {"error": f"Item '{item_id}' not found"})
        raise
    return response(204)


ROUTES = {
    ("GET", "/health"): health,
    ("GET", "/items"): list_items,
    ("POST", "/items"): create_item,
    ("GET", "/items/{id}"): get_item,
    ("PUT", "/items/{id}"): update_item,
    ("DELETE", "/items/{id}"): delete_item,
}


def lambda_handler(event, context):
    method = event.get("httpMethod")
    resource = event.get("resource")
    logger.info("Request: %s %s", method, resource)

    handler = ROUTES.get((method, resource))
    if handler is None:
        return response(404, {"error": f"Route {method} {resource} not found"})

    try:
        return handler(event)
    except ValueError as e:
        return response(400, {"error": str(e)})
    except Exception:
        logger.exception("Unhandled error")
        return response(500, {"error": "Internal server error"})