from datasets import load_dataset
from modelscope import AutoModelForCausalLM, AutoTokenizer


#------------------------- GRPO前模型测试--------------------------
model_name = "./models/Qwen2.5-0.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
) # 加载模型
tokenizer = AutoTokenizer.from_pretrained(model_name) # 加载模型分词器

prompt = "Joy can read 8 pages of a book in 20 minutes. How many hours will it take her to read 120 pages?"
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
) # 将提示词代入Qwen提问模板
model_inputs = tokenizer([text], return_tensors="pt").to(model.device) # 模板文字经过分词器转换成向量形式后转移到GPU上
print(model_inputs)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]
print(generated_ids)

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(response)

#------------------------- 准备数据集--------------------------
data = load_dataset('openai/gsm8k', 'main')
print(data)

#------------------------- 注册wandb--------------------------
import wandb
wandb.login(key="根据笔者教程注册的WandB API KEY")
wandb.init(project="GRPO-test")

#--------------------------GRPO正式训练流程--------------------
import re # 正则表达式匹配库
import torch # 导入pytorch库
from datasets import load_dataset, Dataset # 导入数据集处理依赖库
from transformers import AutoTokenizer, AutoModelForCausalLM # 导入模型加载库
from trl import GRPOConfig, GRPOTrainer # 导入GRPO算法配置和训练类

SYSTEM_PROMPT = """
Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>
"""

XML_COT_FORMAT = """\
<reasoning>
{reasoning}
</reasoning>
<answer>
{answer}
</answer>
"""

# 提取answer标签之间内容
def extract_xml_answer(text):
    answer = text.split("<answer>")[-1]
    answer = answer.split("</answer>")[0]
    return answer.strip()

# 提取gsm8k数据集中的答案部分
def extract_hash_answer(text: str) -> str | None:
    if "####" not in text:
        return None
    return text.split("####")[1].strip()

# 获取gsm8k数据集并处理
def get_gsm8k_questions(split = "train"):
    data = load_dataset('openai/gsm8k', 'main')[split]
    data = data.map(lambda x: {
        'prompt': [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': x['question']}
        ],
        'answer': extract_hash_answer(x['answer'])
    })
    return data
dataset = get_gsm8k_questions()
print(dataset)
print(dataset[0])

# 在GRPO提问前测试在指定系统格式提示词情况下模型回复效果能否改善
messages = dataset[0]['prompt']
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print(response)


# 检查模型输出是否与正确答案匹配，并根据匹配情况返回奖励分数
def correctness_reward_func(prompts, completions, answer, **kwargs):
    responses = [completion[0]['content'] for completion in completions]
    q = prompts[0][-1]['content']
    extracted_responses = [extract_xml_answer(r) for r in responses]
    print('-' * 20, f"Question:\n{q}", f"\nAnswer:\n{answer[0]}", f"\nResponse:\n{responses[0]}",
          f"\nExtracted:\n{extracted_responses[0]}")
    return [2.0 if r == a else 0.0 for r, a in zip(extracted_responses, answer)]


# 检查模型输出是否为有效的整数，并根据结果给予奖励。
def int_reward_func(completions, **kwargs):
    responses = [completion[0]['content'] for completion in completions]
    extracted_responses = [extract_xml_answer(r) for r in responses]
    return [0.5 if r.isdigit() else 0.0 for r in extracted_responses]


# 检查模型的输出是否符合严格的格式要求，包括<reasoning></reasoning>和<answer></answer>标签
def strict_format_reward_func(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"^<reasoning>\n.*?\n</reasoning>\n<answer>\n.*?\n</answer>\n$"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


# 检查模型的输出是否符合稍微宽松的格式要求， <reasoning>和<answer>标签之间可以有空白字符。
def soft_format_reward_func(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<reasoning>.*?</reasoning>\s*<answer>.*?</answer>"
    responses = [completion[0]["content"] for completion in completions]
    matches = [re.match(pattern, r) for r in responses]
    return [0.5 if match else 0.0 for match in matches]


# 计算文本中 和 标签的出现次数，并根据它们的位置和频率分配奖励。
def count_xml(text):
    count = 0.0
    if text.count("<reasoning>\n") == 1:
        count += 0.125
    if text.count("\n</reasoning>\n") == 1:
        count += 0.125
    if text.count("\n<answer>\n") == 1:
        count += 0.125
        count -= len(text.split("\n</answer>\n")[-1]) * 0.001
    if text.count("\n</answer>") == 1:
        count += 0.125
        count -= (len(text.split("\n</answer>")[-1]) - 1) * 0.001
    return count


# 该函数用于计算每个模型输出的 XML 结构符合度，并返回奖励分数
def xmlcount_reward_func(completions, **kwargs):
    contents = [completion[0]["content"] for completion in completions]
    return [count_xml(c) for c in contents]

model_name = "models/Qwen2.5-0.5B-Instruct"
output_dir="outputs/Qwen-0.5B-GRPO"
run_name="Qwen-0.5B-GRPO-gsm8k"

training_args = GRPOConfig(
    output_dir=output_dir,
    run_name=run_name,
    learning_rate=5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type='cosine',
    logging_steps=1,
    bf16=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_generations=16,
    max_prompt_length=256,
    max_completion_length=200,
    num_train_epochs=1,
    save_steps=100,
    max_grad_norm=0.1,
    log_on_each_node=False,
    use_vllm=False,
    vllm_gpu_memory_utilization=.3,
    vllm_device="cuda:0",
    report_to="wandb"
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map=None
).to("cuda")

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        xmlcount_reward_func,
        soft_format_reward_func,
        strict_format_reward_func,
        int_reward_func,
        correctness_reward_func],
    args=training_args,
    train_dataset=dataset,
)

trainer.train()

#------------------------------模型测试---------------------------
grpo_model_name = "./outputs/Qwen-0.5B-GRPO"

model = AutoModelForCausalLM.from_pretrained(
    grpo_model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = "Joy can read 8 pages of a book in 20 minutes. How many hours will it take her to read 120 pages?"
messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": prompt}
]

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)[0]

print(response)

