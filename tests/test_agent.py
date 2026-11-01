"""Tests for the agent, using a fake Bedrock client so no AI calls are made."""

import copy
import json
import os
import sys

import boto3
import pytest
from moto import mock_aws

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

TABLE = "items-test"


class FakeBedrock:
    """Returns scripted model replies in order and records what it was sent."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.calls = []

    def converse(self, **kwargs):
        self.calls.append(copy.deepcopy(kwargs))
        return self.replies.pop(0)


def tool_use(name, tool_input, tool_id="t1"):
    return {
        "stopReason": "tool_use",
        "output": {"message": {"role": "assistant", "content": [
            {"toolUse": {"toolUseId": tool_id, "name": name, "input": tool_input}}
        ]}},
    }


def final(text):
    return {
        "stopReason": "end_turn",
        "output": {"message": {"role": "assistant", "content": [{"text": text}]}},
    }


@pytest.fixture
def agent(monkeypatch):
    monkeypatch.setenv("TABLE_NAME", TABLE)
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    with mock_aws():
        boto3.client("dynamodb").create_table(
            TableName=TABLE,
            KeySchema=[{"AttributeName": "id", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "id", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        from items import app as items_app
        from agent import app as module
        items_app._table = None
        yield module


def invoke(agent, prompt):
    res = agent.lambda_handler({"body": json.dumps({"prompt": prompt})}, None)
    return res["statusCode"], json.loads(res["body"])


def test_agent_creates_then_lists(agent):
    agent._bedrock = FakeBedrock([
        tool_use("create_item", {"name": "Laptop", "attributes": {"price": 999.5}}, "t1"),
        tool_use("list_items", {}, "t2"),
        final("Added Laptop. You now have 1 item."),
    ])
    status, body = invoke(agent, "Add a laptop for $999.50 and list everything")

    assert status == 200
    assert body["answer"] == "Added Laptop. You now have 1 item."
    assert [s["tool"] for s in body["steps"]] == ["create_item", "list_items"]

    # The list_items result sent back to the model contains the new item
    last_call = agent._bedrock.calls[-1]
    tool_result = last_call["messages"][-1]["content"][0]["toolResult"]
    assert tool_result["content"][0]["json"]["items"][0]["price"] == 999.5


def test_agent_reports_tool_errors(agent):
    agent._bedrock = FakeBedrock([
        tool_use("update_item", {"id": "missing", "fields": {"price": 1}}),
        final("I couldn't find that item."),
    ])
    status, body = invoke(agent, "Set price of item 'missing' to 1")
    assert status == 200
    sent = agent._bedrock.calls[-1]["messages"][-1]["content"][0]["toolResult"]
    assert "not found" in sent["content"][0]["json"]["error"]


def test_agent_step_limit(agent, monkeypatch):
    monkeypatch.setattr(agent, "MAX_STEPS", 2)
    agent._bedrock = FakeBedrock([tool_use("list_items", {}), tool_use("list_items", {})])
    status, body = invoke(agent, "loop forever")
    assert status == 200 and "step limit" in body["answer"]


def test_agent_validation(agent):
    assert agent.lambda_handler({"body": "{}"}, None)["statusCode"] == 400
    assert agent.lambda_handler({"body": "{bad"}, None)["statusCode"] == 400