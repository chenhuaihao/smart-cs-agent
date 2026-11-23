"""
app/llm/__init__.py

模型适配层（端口-适配器模式）：
  - base.py     端口：BaseLLMProvider 抽象 + 统一流式输出结构
  - deepseek.py 适配器：基于 openai.AsyncOpenAI 的 OpenAI 兼容实现
  - factory.py  工厂：按 LLM_PROVIDER 环境变量返回 Provider 单例
"""
