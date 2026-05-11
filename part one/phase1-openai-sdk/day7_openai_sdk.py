"""
第7天：OpenAI Python SDK 实战
===============================
完成这个文件，你就掌握了 LLM 应用开发最核心的技能——调 API。

三个 provider，一套 SDK：
  - MINI MAX    → https://api.minimaxi.com/v1
  - DeepSeek (V3/R1)    → https://api.deepseek.com/v1
  - KIMI     → https://api.moonshot.cn/v1 

它们都兼容 OpenAI 接口格式，改 base_url + api_key 即可切换。
"""

import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # 从 .env 文件加载 API Key

# ============================================================
# 1. 基础调用：给模型发一句话，拿回一句话
# ============================================================

def demo_1_basic_call():
    """最简调用——就是一问一答"""
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )

    response = client.chat.completions.create(
        model="deepseek-chat",  # 即 DeepSeek-V3
        messages=[
            {"role": "system", "content": "你是一个 Java 转 Python 的编程助手，回答简洁直接。"},
            {"role": "user", "content": "Python 的 async/await 和 Java 的 CompletableFuture 有什么关键区别？"},
        ],
    )

    print(response.choices[0].message.content)


# ============================================================
# 2. 流式输出（SSE）：一个字一个字往外蹦
# ============================================================

def demo_2_streaming():
    """
    流式输出——用户体验的关键。
    对比：普通调用 = 等 5 秒 → 一次性出结果
         流式调用 = 0.5 秒开始出字 → 打字机效果
    """
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )

    stream = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": "用 100 字介绍什么是 RAG（检索增强生成）。"},
        ],
        stream=True,
    )

    print("模型回复: ", end="", flush=True)
    for chunk in stream:
        delta = chunk.choices[0].delta
        if delta.content:
            print(delta.content, end="", flush=True)
    print()  # 换行


# ============================================================
# 3. Function Calling / Tool Use：让模型调用你的函数
# ============================================================

def get_weather(city: str) -> str:
    """模拟天气查询——实际项目里这会调真实 API"""
    weather_data = {
        "北京": "晴，18°C，北风3级",
        "上海": "多云，22°C，东南风2级",
        "深圳": "阵雨，26°C，南风4级",
    }
    return weather_data.get(city, f"未找到{city}的天气数据")


def demo_3_function_calling():
    """
    Function Calling 是 LLM 应用的灵魂：
    模型不直接输出文本，而是输出一个 JSON → 你的代码执行 → 把结果还给模型。
    这就是 Agent 的底层机制。
    """
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "查询指定城市的天气",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {
                            "type": "string",
                            "description": "城市名称，如 '北京'、'上海'",
                        }
                    },
                    "required": ["city"],
                },
            },
        }
    ]

    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "user", "content": "北京今天天气怎么样？"},
        ],
        tools=tools,
    )

    # 模型返回的不是文本，而是一个 tool_call
    msg = response.choices[0].message
    if msg.tool_calls:
        tool_call = msg.tool_calls[0]
        func_name = tool_call.function.name
        args = eval(tool_call.function.arguments)  # 生产环境用 json.loads

        print(f"模型想调用: {func_name}({args})")

        # 执行函数
        result = get_weather(**args)
        print(f"函数返回: {result}")

        # 把结果还给模型，让它用自然语言组织回答
        final = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "user", "content": "北京今天天气怎么样？"},
                {"role": "assistant", "tool_calls": [tool_call]},
                {"role": "tool", "tool_call_id": tool_call.id, "content": result},
            ],
        )
        print(f"模型最终回答: {final.choices[0].message.content}")
    else:
        print(f"模型直接回答: {msg.content}")


# ============================================================
# 4. 多 Provider 切换：同一个接口，换三套参数
# ============================================================

def demo_4_multi_provider():
    """
    用同一段对话，分别调三个模型，对比效果。
    这是实际项目里的模型选型思维——不同任务挑不同模型。
    """
    prompt = "把'Hello World'翻译成中文"

    providers = {
        "DeepSeek-V3": {
            "api_key": os.getenv("DEEPSEEK_API_KEY"),
            "base_url": "https://api.deepseek.com",
            "model": "deepseek-chat",
        },
        "kimi-k2.6": {
            "api_key": os.getenv("KIMI_API_KEY"),
            "base_url": "https://api.moonshot.cn/v1",
            "model": "kimi-k2.6",
        },
        "MiniMax-M2.7": {
            "api_key": os.getenv("MINIMAX_API_KEY"),
            "base_url": "https://api.minimaxi.com/v1",
            "model": "MiniMax-M2.7",
        },
    }

    for name, config in providers.items():
        if not config["api_key"]:
            print(f"⏭ {name}: 未配置 API Key，跳过")
            continue

        client = OpenAI(api_key=config["api_key"], base_url=config["base_url"])
        response = client.chat.completions.create(
            model=config["model"],
            messages=[{"role": "user", "content": prompt}],
        )
        print(f"🤖 {name}: {response.choices[0].message.content}")


# ============================================================
# 5. 结构化输出：让模型按你指定的 JSON Schema 输出
# ============================================================

def demo_5_structured_output():
    """
    生产环境中你几乎总是希望模型输出结构化数据，
    而不是自由文本。Function Calling 也是一种结构化输出方式，
    但如果你只是想要 JSON，用 response_format 更简单。
    """
    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )

    # 注意：DeepSeek 也支持这个参数，但某些模型可能需要 json_object
    response = client.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": "你只输出 JSON，不要加任何解释。"},
            {"role": "user", "content": "分析这句话的情感：'不好不坏'"},
        ],
        response_format={"type": "json_object"},
    )

    import json
    result = json.loads(response.choices[0].message.content)
    print(f"结构化结果: {json.dumps(result, ensure_ascii=False, indent=2)}")


# ============================================================
# 运行入口
# ============================================================

if __name__ == "__main__":
    import sys

    demos = {
        "1": ("基础调用", demo_1_basic_call),
        "2": ("流式输出", demo_2_streaming),
        "3": ("Function Calling", demo_3_function_calling),
        "4": ("多Provider对比", demo_4_multi_provider),
        "5": ("结构化输出", demo_5_structured_output),
    }

    if len(sys.argv) > 1 and sys.argv[1] in demos:
        name, fn = demos[sys.argv[1]]
        print(f"\n{'='*60}")
        print(f"  Demo {sys.argv[1]}: {name}")
        print(f"{'='*60}\n")
        fn()
    else:
        print("用法: python day7_openai_sdk.py <编号>")
        print()
        for k, (name, _) in demos.items():
            print(f"  {k} - {name}")
        print("\n示例: python day7_openai_sdk.py 1")
        print("\n运行前请先创建 .env 文件并填入 API Key：")
        print("  DEEPSEEK_API_KEY=sk-xxxxx")
