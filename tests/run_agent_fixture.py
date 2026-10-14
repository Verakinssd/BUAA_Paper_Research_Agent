"""Exercise the real CLI/MCP with test-only network/model substitutes."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from support import FakeTaskAgent, sample_papers, fake_download


async def retrieve_for_test(query, limit=8, concurrent=False):
    return sample_papers(), {name: {"status": "ok", "count": count}
                             for name, count in (("arxiv", 2), ("openalex", 1))}


async def run_main(args):
    import main_agent

    class ToolModel:
        def __init__(self):
            self.calls = 0

        async def chat(self, messages, tools=None):
            await asyncio.sleep(float(os.getenv("TEST_AGENT_DELAY", "0")))
            self.calls += 1
            if self.calls == 1:
                name, arguments = "search_papers", {"query": "research paper retrieval"}
            else:
                name = "build_report"
                arguments = dict(run_id=json.loads(messages[-1]["content"])["run_id"],
                                 idea=args.idea, top_k=args.top_k)
            return {"role": "assistant", "content": None, "tool_calls": [
                {"id": f"test-call-{self.calls}", "type": "function", "function": {
                    "name": name, "arguments": json.dumps(arguments)}}]}

    original = main_agent.StdioServerParameters

    def test_server(**kwargs):
        kwargs["args"] = [str(Path(__file__).resolve()), "--server"] + kwargs["args"][1:]
        return original(**kwargs)

    with patch.object(main_agent, "LLM", ToolModel), \
            patch.object(main_agent, "StdioServerParameters", test_server):
        await main_agent.run(args)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", action="store_true")
    parser.add_argument("--idea", default="test")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--concurrent", action="store_true")
    args = parser.parse_args()
    if args.server:
        import server
        os.environ["PAPER_CACHE_DIR"] = str(Path(os.environ["PAPER_OUTPUT_ROOT"]) / "_test_cache")
        server.config.update(concurrent=args.concurrent)
        with patch.object(server, "retrieve", retrieve_for_test), \
                patch.object(server, "TaskAgent", FakeTaskAgent), \
                patch("research.pipeline.download_pdf", fake_download):
            server.mcp.run(transport="stdio")
    else:
        asyncio.run(run_main(args))
