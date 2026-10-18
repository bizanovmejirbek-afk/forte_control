"""Локальная мультимодальная модель (VLM).

VLM_BACKEND=none  — заглушка: описание по шаблону (для разработки без модели)
VLM_BACKEND=mlx   — Qwen2.5-VL-7B (4 бит) через mlx-vlm на Mac с Apple Silicon. Всё локально.
"""
import os

from .. import config


class TemplateVLM:
    name = "template"

    def review(self, candidate, frames, rules):
        t = rules["types"].get(candidate.type_id, rules["types"]["other"])
        desc = t["template"].format(
            objects=", ".join(candidate.objects) or "ценности",
            description=candidate.note or "Наблюдается действие, нехарактерное для обычной работы кассира.",
        )
        return True, candidate.type_id, desc

    def ask(self, image_paths, prompt, max_tokens=64):
        return ""


class MlxVLM:
    """Обёртка над mlx-vlm. Поддерживает старый и новый API (generate -> str или объект с .text)."""

    def __init__(self, model_id):
        os.environ.setdefault("HF_HUB_OFFLINE", "1")  # после скачивания весов интернет не нужен
        from mlx_vlm import generate, load
        from mlx_vlm.prompt_utils import apply_chat_template
        self._generate, self._template = generate, apply_chat_template
        self.model, self.processor = load(model_id)
        self.config = getattr(self.model, "config", None)
        try:
            from mlx_vlm.utils import load_config
            self.config = load_config(model_id)
        except Exception:
            pass
        self.name = "mlx:" + model_id.split("/")[-1]

    def ask(self, image_paths, prompt, max_tokens=64):
        formatted = self._template(self.processor, self.config, prompt, num_images=len(image_paths))
        try:
            out = self._generate(self.model, self.processor, formatted, image_paths,
                                 max_tokens=max_tokens, temperature=0.0, verbose=False)
        except TypeError:  # старые версии без temperature
            out = self._generate(self.model, self.processor, formatted, image_paths,
                                 max_tokens=max_tokens, verbose=False)
        return (getattr(out, "text", out) or "").strip()

    def review(self, candidate, frames, rules):
        t = rules["types"].get(candidate.type_id, rules["types"]["other"])
        return True, candidate.type_id, candidate.note or t["template"]


def get_vlm():
    backend = config.VLM_BACKEND
    if backend == "none":
        return TemplateVLM()
    if backend == "mlx":
        return MlxVLM(os.getenv("VLM_MODEL", "mlx-community/Qwen2.5-VL-7B-Instruct-4bit"))
    raise ValueError(f"Неизвестный VLM_BACKEND={backend}")
