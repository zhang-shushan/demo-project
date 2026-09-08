from pydantic import SecretStr

TONGYI_MODEL = "qwen3.7-flash"

TONGYI_API_KEY = SecretStr("你的tongyi大模型API Key")

TONGYI_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"

TONGYI_TEMPERATURE = 0.3

TONGYI_MAX_OUTPUT_LENGTH = 1024