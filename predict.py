from cog import BasePredictor, Path, Input
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
from pget import pget_manifest
import subprocess


class Predictor(BasePredictor):

    def setup(self):
        pget_manifest()
        self.device = "cuda"
        self.tokenizer = AutoTokenizer.from_pretrained("./weights", trust_remote_code=True)
        self.model = (
            AutoModel.from_pretrained("./weights",
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            .to(self.device)
            .eval()
        )

    def predict(self,
        image: Path = Input(description="Image input"),
        prompt: str = Input(description="Prompt"),
        max_length: int = Input(description="Maximum number of tokens to generate.", default=512, ge=1, le=8192),
        top_k: int = Input(description="Top-K sampling", default=1, ge=1, le=1000)
    ) -> str:
        print(subprocess.check_output(["nvidia-smi"]).decode("utf-8"))
        image = Image.open(image).convert("RGB")
        inputs = self.tokenizer.apply_chat_template(
            [{"role": "user", "image": image, "content": prompt}],
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            return_dict=True,
            max_length=max_length,
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_length=max_length, do_sample=True, top_k=top_k)
            outputs = outputs[:, inputs["input_ids"].shape[1] :]
            return self.tokenizer.decode(outputs[0])
