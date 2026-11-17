"""Photo generation: Stable Diffusion + LLM prompt engineering + LLM assessment loop. Port of ImageGenerator.cpp and
StableDiffusionClientImpl.cpp."""

from __future__ import annotations

import base64
import io
import json
import logging
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
from PIL import Image

from .config import config
from .llm import ROLE_USER, Message, OpenAIChat, Params, Session, embed_image
from .prompts import prompts

log = logging.getLogger("ImageGenerator")

TRIAL_COUNT = 10

# Minimize the risk of generating too young/child characters. This is a safety precaution for the instance owner since
# CP is prohibited in many jurisdictions. Adjusts the prompts so the ImageGenerator won't make up children.
PREVENT_GENERATING_CHILDREN = True


class StableDiffusionClient:
    """AUTOMATIC1111 WebUI API (`--api`)."""

    def __init__(self, http: httpx.AsyncClient | None = None) -> None:
        self._http = http

    async def _post(self, path: str, body: dict | None = None) -> httpx.Response:
        ep = config().sdEndpoint
        headers = {"Content-Type": "application/json"}
        if ep.bearerKey:
            headers["Authorization"] = f"Bearer {ep.bearerKey}"
        client = self._http or httpx.AsyncClient(timeout=float(config().requestTimeoutSecs))
        try:
            return await client.post(ep.baseUrl + path, json=body, headers=headers)
        finally:
            if self._http is None:
                await client.aclose()

    async def txt2img(self, **req) -> list[Image.Image]:
        body = {
            "prompt": "", "negative_prompt": "", "styles": [], "seed": -1, "subseed": -1, "subseed_strength": 0,
            "seed_resize_from_h": -1, "seed_resize_from_w": -1, "sampler_name": "DPM++ 2M", "scheduler": "Automatic",
            "override_settings": {"sd_model_checkpoint": config().sdCheckpoint}, "batch_size": 1, "n_iter": 1,
            "steps": 50, "cfg_scale": 2, "width": 512, "height": 512, "send_images": True, "save_images": False,
            "enable_hr": False, "hr_scale": 2.0, "hr_upscaler": "Latent", "hr_second_pass_steps": 0,
            "denoising_strength": 1.0,
        }
        body.update(req)
        resp = await self._post("sdapi/v1/txt2img", body)
        data = resp.json()
        return [Image.open(io.BytesIO(base64.b64decode(b))) for b in data.get("images", [])]

    async def unload_checkpoint(self) -> None:
        await self._post("sdapi/v1/unload-checkpoint")


@dataclass
class GalleryImage:
    image: Image.Image
    path: Path


@dataclass
class PromptPair:
    positive: str = ""
    negative: str = ""


@dataclass
class Assessment:
    satisfied: bool
    feedback: str


def parse_response(content: str) -> dict:
    # Basic JSON extraction if the model wrapped it in markdown
    if "```json" in content:
        content = content.split("```json")[1].split("```")[0]
    elif "```" in content:
        content = content.split("```")[1].split("```")[0]
    return json.loads(content)


class ImageGenerator:
    def __init__(self, sd: StableDiffusionClient, llm: OpenAIChat, chat_params: Params) -> None:
        self.sd, self.llm, self.chat_params = sd, llm, chat_params
        self.rng = random.Random()

    def _params(self) -> Params:
        return Params(system_prompt=self.chat_params.system_prompt, config=self.chat_params.config,
                      max_output_tokens=self.chat_params.max_output_tokens)

    async def generate(self, description: str) -> GalleryImage:
        log.debug("generate: %s", description)
        trial = 0
        first_feedback = ""
        while True:  # naxyi: (re)engineer the prompt from scratch
            log.info("Engineering initial prompt for: %s", description)
            current = PromptPair(positive="", negative="(text:2), (signature:2), raw photo")
            await self.engineer_prompt(current, description, prompts().character_appearance)
            full_description = (
                f'<character name="{config().characterName}" canonical_description>\n{prompts().character_appearance}\n'
                f'</character canonical_description>\n<user_description overrides_canonical="true">\n{description}\n'
                '</user_description overrides_canonical="true">')
            log.info("positive=%s\n\nnegative=%s", current.positive, current.negative)
            reengineer = False
            while trial <= TRIAL_COUNT:
                try:
                    trial += 1
                    log.info("Iteration %d", trial)
                    try:
                        images = await self.sd.txt2img(
                            prompt=current.positive, negative_prompt=current.negative, steps=30,
                            cfg_scale=self.rng.uniform(1.0, 5.0), width=self.rng.randint(768, 1400),
                            height=self.rng.randint(768, 1400), enable_hr=True, hr_scale=1.5, hr_upscaler="Latent",
                            hr_second_pass_steps=10, denoising_strength=0.7)
                    except Exception as e:  # noqa: BLE001
                        log.error("Stable diffusion failed: %s", e)
                        images = None
                    if images is not None:
                        if not images:
                            raise RuntimeError("Stable Diffusion returned no images")
                        last = images[0]
                        last.save("image_generator_tmp.png")
                        try:  # unload SD checkpoint after generation
                            await self.sd.unload_checkpoint()
                            log.info("Checkpoint unloaded from VRAM")
                        except Exception as e:  # noqa: BLE001
                            log.warning("Failed to unload checkpoint: %s", e)
                        log.info("Assessing image...")
                        assessment = await self.assess_image(last, full_description)
                        if assessment.satisfied:
                            log.info("Satisfied with the result. %s", assessment.feedback)
                            dst = Path("data") / "gallery" / f"{time.strftime('%Y-%m-%d_%H-%M-%S')}_{time.time_ns() % 10**6}.png"
                            dst.parent.mkdir(parents=True, exist_ok=True)
                            last.save(dst)
                            return GalleryImage(last, dst.absolute())
                        if not first_feedback:
                            first_feedback = assessment.feedback
                        log.info("Not satisfied. Feedback: %s", assessment.feedback)
                    if trial % int(math.sqrt(TRIAL_COUNT)) == 0:
                        log.info("Last trial failed. Retrying with different prompt...")
                        reengineer = True
                        break
                except Exception as e:  # noqa: BLE001
                    log.error("Failed to generate image: %s", e)
            if not reengineer:
                break
        raise RuntimeError(f'can\'t find image: feedback: "{first_feedback}"; make photo_desc shorter')

    async def engineer_prompt(self, out: PromptPair, description: str, appearance_prompt: str, feedback: str = "") -> None:
        log.debug("engineerPrompt description=%s feedback=%s", description, feedback)
        params = self._params()
        # NOTE: the description is passed to the prompt engineer as is. (The C++ kernel stripped some words before
        # this call and re-appended them to the Stable Diffusion prompt afterwards, hiding them from the LLM;
        # that content-filter workaround is intentionally not ported.)
        params.system_prompt = (prompts().image_engineer_system
                                .replace("{CHARACTER_NAME}", config().characterName)
                                .replace("{CHARACTER_APPEARANCE_PROMPT}", appearance_prompt)
                                .replace("{SANITIZED_PHOTO_DESCRIPTION}", description))
        message = "# Previous prompt iteration \n"
        message += f"Positive prompt: {out.positive}\n\n"
        message += f"Negative prompt: {out.negative}\n\n"
        if feedback:
            message += "# Feedback to previous prompt iteration\n" + feedback
        message += "\n\n" + prompts().image_engineer_instructions
        message += "\nGenerate SD prompt:"
        messages = Session([Message(role=ROLE_USER, content=message)])

        response = await self.llm.chat(params, messages)
        while True:
            if not response.choices:
                raise RuntimeError("OpenAI returned no choices for initial prompt engineering")
            content = response.choices[0].message.content
            try:
                data = parse_response(content)
                out.positive, out.negative = str(data["positivePrompt"]), str(data["negativePrompt"])
            except json.JSONDecodeError:
                response = await self.llm.chat(params, messages)  # truncated/garbled JSON: ask again
                continue
            too_long = None
            for name, text in (("positive", out.positive), ("negative", out.negative)):
                text = text.replace(") ", "), ")  # add commas
                if name == "positive":
                    out.positive = text
                else:
                    out.negative = text
                if text.count(" ") > 60:  # long prompts distort the character base design.
                    too_long = name
                    break
            if too_long is None:
                break
            if len(messages) > 3:
                raise RuntimeError(f"adjusted {too_long} prompt is too long.")
            messages << Message(role=ROLE_USER, content=f"Adjusted {too_long} prompt is too long. Shorten it to 50 words "
                                                        "or less.; restructure or adjust word (weights:1.5) instead")
            response = await self.llm.chat(params, messages)
        if PREVENT_GENERATING_CHILDREN:
            out.negative += " child"

    async def assess_image(self, image: Image.Image, description: str) -> Assessment:
        params = self._params()
        # Note: chat_params.config should ideally be a vision-capable model.
        params.system_prompt = prompts().image_assess_system + description
        if PREVENT_GENERATING_CHILDREN:
            params.system_prompt += "\nThe character(s) must not appear as child"
        messages = Session([Message(role=ROLE_USER, content="Assess this image: " + embed_image(image))])
        for _ in range(3):
            response = await self.llm.chat(params, messages)
            if not response.choices:
                raise RuntimeError("OpenAI returned no choices for image assessment")
            content = response.choices[0].message.content
            try:
                data = parse_response(content)
                return Assessment(bool(data["satisfied"]), str(data["feedback"]))
            except (json.JSONDecodeError, KeyError) as e:
                log.error("Failed to parse assessment JSON: %s\nContent: %s", e, content)
        # Fallback: assume not satisfied if parsing fails, to avoid accepting garbage
        return Assessment(False, "")
