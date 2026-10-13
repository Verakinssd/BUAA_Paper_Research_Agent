"""Check local model listing, text, auto tool calling and JSON before a full run."""
import asyncio
import json
from dotenv import load_dotenv
import httpx
from research.llm import LLM, parse_json


async def run():
    load_dotenv()
    llm = LLM()
    async with httpx.AsyncClient(timeout=30, trust_env=llm.trust_env) as client:
        response = await client.get(llm.base + "/models", headers=llm.headers())
        response.raise_for_status()
    names = [m["id"] for m in response.json()["data"]]
    if llm.model not in names:
        raise ValueError(f"LLM_MODEL 不在服务返回列表中：{names}")
    print("模型服务与模型名称：通过", flush=True)
    await llm.text("你是简洁的助手。", "只回复‘本地模型已连接’。")
    print("文本生成与上下文检查：通过", flush=True)
    tools = [{"type": "function", "function": {
        "name": "get_demo_paper_count", "description": "查询合成演示中论文的数量，不能凭空猜测数量。",
        "parameters": {"type": "object", "properties": {}, "required": []},
    }}]
    messages = [{"role": "system", "content": "你是工具助手，必须调用提供的查询工具取得答案。"},
                {"role": "user", "content": "查询演示中论文数量。"}]
    result = await llm.chat(messages, tools)
    calls = result.get("tool_calls") or []
    if not calls or any(c["function"]["name"] != "get_demo_paper_count" for c in calls):
        raise ValueError("没有解析出预期工具调用；llama.cpp 请检查 --jinja/模型模板，vLLM 请检查工具解析器")
    messages.append(result)
    for call in calls:
        arguments = json.loads(call["function"]["arguments"])
        if arguments:
            raise ValueError("测试工具参数不正确")
        messages.append({"role": "tool", "tool_call_id": call["id"], "content": '{"count":3}'})
    reply = await llm.chat(messages)
    if not reply.get("content"):
        raise ValueError("工具结果回传后没有文本输出")
    print("自动工具调用与结果回传：通过", flush=True)
    value = parse_json(await llm.text("仅输出 JSON，不写其他文本。", '返回对象 {"ready":true}。'))
    if value.get("ready") is not True:
        raise ValueError("模型 JSON 格式检查未通过")
    print("JSON 输出：通过。现在可以运行 main_agent.py。", flush=True)


if __name__ == "__main__":
    asyncio.run(run())
