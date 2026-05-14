from transformers import AutoModelForImageSegmentation
import torch
from torchvision import transforms
from PIL import Image


class BiRefNet:
    def __init__(
        self,
        model_name: str = "ZhengPeng7/BiRefNet",
        fallback_model_name: str = "ZhengPeng7/BiRefNet",
        allow_opaque_fallback: bool = False,
        lazy_load: bool = False,
    ):
        self.model_name = model_name
        self.fallback_model_name = fallback_model_name
        self.allow_opaque_fallback = allow_opaque_fallback
        self.lazy_load = lazy_load
        self.model = None
        self._device = torch.device("cpu")
        self._load_error = None
        self.transform_image = transforms.Compose(
            [
                transforms.Resize((1024, 1024)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        if not self.lazy_load:
            self._load_model()

    def _candidate_models(self) -> list[str]:
        candidates = []
        for value in (self.model_name, self.fallback_model_name):
            if value and value not in candidates:
                candidates.append(value)
        return candidates

    def _load_model(self) -> bool:
        if self.model is not None:
            return True
        errors = []
        for candidate in self._candidate_models():
            try:
                print(f"[RMBG] Loading background remover: {candidate}", flush=True)
                model = AutoModelForImageSegmentation.from_pretrained(
                    candidate, trust_remote_code=True
                )
                model.eval()
                model.to(self._device)
                self.model = model
                print(f"[RMBG] Ready: {candidate}", flush=True)
                return True
            except Exception as error:
                errors.append(f"{candidate}: {error}")
                print(f"[RMBG] Could not load {candidate}: {error}", flush=True)
        self._load_error = "\n".join(errors)
        if self.allow_opaque_fallback:
            print(
                "[RMBG] Background remover unavailable; continuing with opaque-alpha input image.",
                flush=True,
            )
            return False
        raise RuntimeError(self._load_error or "Background remover could not be loaded.")

    def to(self, device: str):
        self._device = torch.device(device)
        if self.model is not None:
            self.model.to(self._device)
        return self

    def cuda(self):
        target = "cuda" if torch.cuda.is_available() else self._device
        return self.to(target)

    def cpu(self):
        return self.to("cpu")
        
    def __call__(self, image: Image.Image) -> Image.Image:
        image_size = image.size
        rgb_image = image.convert("RGB")
        if not self._load_model():
            return rgb_image.convert("RGBA")
        input_images = self.transform_image(rgb_image).unsqueeze(0).to(self._device)
        # Prediction
        with torch.no_grad():
            preds = self.model(input_images)[-1].sigmoid().cpu()
        pred = preds[0].squeeze()
        pred_pil = transforms.ToPILImage()(pred)
        mask = pred_pil.resize(image_size)
        output = rgb_image.convert("RGBA")
        output.putalpha(mask)
        return output
