import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from mlx_vlm.utils import load_config


# Load the model

class QwenImageModel:

    def __init__(self):
        model_path = "mlx-community/Qwen2-VL-2B-Instruct-4bit"
        self.model, self.processor = load(model_path)
        self.config = load_config(model_path)

    def single(self, prompt, image_path):
        formatted_prompt = apply_chat_template(
            self.processor, self.config, prompt, num_images=1
        )
        return generate(self.model, self.processor, formatted_prompt, [image_path],
                        verbose=False)
