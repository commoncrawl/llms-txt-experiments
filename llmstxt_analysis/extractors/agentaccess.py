"""Track B (adjunct) — agent-access advertising.

Several generators use llms.txt not as a link index but as a service-discovery
channel, pointing agents at an MCP endpoint. Cheap to detect and it explains a
large share of the generator distribution, so it is carried alongside track B.
"""
from __future__ import annotations

import re

from ..record import Record
from ..registry import BOOL, I32, STR, RowExtractor, register

_MCP_ENDPOINT = re.compile(r"https?://[^\s\"'<>)\]]*?/(?:_api/)?mcp\b[^\s\"'<>)\]]*", re.I)
_MCP_MENTION = re.compile(r"model\s*context\s*protocol|\bMCP\b|/mcp\b", re.I)
_TOOL_HEADING = re.compile(r"^#{3,4}\s+([A-Z][A-Za-z0-9_]{2,40})\s*$", re.M)


@register
class AgentAccessExtractor(RowExtractor):
    NAME = "agent_access"
    TRACK = "B"
    FIELDS = {
        "mentions_mcp": BOOL,
        "mcp_endpoint": STR,
        "n_tool_headings": I32,
    }

    def extract(self, rec: Record) -> dict:
        head = rec.head
        ep = _MCP_ENDPOINT.search(head)
        return {
            "mentions_mcp": bool(_MCP_MENTION.search(head)),
            "mcp_endpoint": (ep.group(0)[:200] if ep else ""),
            "n_tool_headings": len(_TOOL_HEADING.findall(head)),
        }
