import torch
import math
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import logging

logger = logging.getLogger(__name__)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import logging

# 假设 logger 已经定义
logger = logging.getLogger(__name__)

class PPLEngine:
    def __init__(self, config, args):
        gpu_id = args.gpu_id
        self.model_name = config["model"]["name"]

        if torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if gpu_id >= num_gpus:
                logger.warning(
                    f"Configured GPU ID {gpu_id} not found (only {num_gpus} available). "
                    f"Fallback to cuda:0"
                )
                gpu_id = 0
            device_str = f"cuda:{gpu_id}"
        else:
            device_str = "cpu"

        logger.info(f"Loading model: {self.model_name} on {device_str}")
        self.device = torch.device(device_str)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )

        dtype = torch.float16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            cache_dir="./HF_Cache",
            torch_dtype=dtype,
            trust_remote_code=True
        ).to(self.device)

        self.model.eval()

    def compute_conditional_ppl(self, context: str, target: str) -> float:
        """
        计算 PPL(target | context)
        """
        if not target.strip():
            return 0.0

        if context.strip():
            full_text = context + "\n" + target
        else:
            full_text = target

        inputs = self.tokenizer(
            full_text,
            return_tensors="pt",
            add_special_tokens=True
        ).to(self.device)

        input_ids = inputs.input_ids  # [1, T]

        if context.strip():
            context_ids = self.tokenizer(
                context + "\n",
                return_tensors="pt",
                add_special_tokens=False
            ).input_ids
            context_len_tokens = context_ids.shape[1]

            if (
                self.tokenizer.bos_token_id is not None
                and input_ids[0, 0] == self.tokenizer.bos_token_id
            ):
                context_len_tokens += 1
        else:
            context_len_tokens = 0

        labels = input_ids.clone()
        if context_len_tokens > 0:
            labels[:, :context_len_tokens] = -100  # HF ignore index

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, labels=labels)
            loss = outputs.loss  

        if loss is None or torch.isnan(loss):
            return float("inf")

        # PPL = exp(mean loss)
        try:
            return math.exp(loss.item())
        except OverflowError:
            return float("inf")
