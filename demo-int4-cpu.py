from transformers import AutoTokenizer, AutoModel
import time
from loguru import logger


logger.debug(f'开始加载模型')
tokenizer = AutoTokenizer.from_pretrained(
    "THUDM/chatglm2-6b-int4", trust_remote_code=True)
model = AutoModel.from_pretrained(
    "THUDM/chatglm2-6b-int4", trust_remote_code=True).float()
model = model.eval()

logger.debug(f'模型加载完毕')

# s=time.time()
# response, history = model.chat(tokenizer, "你好", history=[])
# print(response)
# e=time.time()
# logger.debug(f'pay time is {round(e-s,2)} 秒')

s=time.time()
response, history = model.chat(tokenizer, "晚上睡不着应该怎么办")
print(response)
e=time.time()
logger.debug(f'pay time is {round(e-s,2)} 秒')

s=time.time()
response, history = model.chat(tokenizer, "三国时期是公元多少年")
print(response)
e=time.time()
logger.debug(f'pay time is {round(e-s,2)} 秒')