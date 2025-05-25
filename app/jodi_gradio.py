import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["GRADIO_TEMP_DIR"] = "./tmp"

import argparse
import gradio as gr
from PIL import Image
from typing import Any
from pathlib import Path

import torch
import torchvision.transforms as T

from app.jodi_pipeline import JodiPipeline
from data.datasets.jodi_dataset import JodiDataset
from model.postprocess import (
    ImagePostProcessor, LineartPostProcessor, EdgePostProcessor, DepthPostProcessor,
    NormalPostProcessor, AlbedoPostProcessor, SegADE20KPostProcessor, OpenposePostProcessor,
)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="./configs/inference.yaml")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model checkpoint file.")
    return parser


def get_closest_ratio(height: float, width: float, ratios: dict):
    aspect_ratio = height / width
    closest_ratio = min(ratios.keys(), key=lambda ratio: abs(float(ratio) - aspect_ratio))
    return ratios[closest_ratio], float(closest_ratio)


def change_height_width_by_ar(ar):
    height, width = JodiDataset.aspect_ratio[ar]
    height, width = int(height), int(width)
    return height, width


def detect_aspect_ratio_from_image(image):
    (height, width), ratio = get_closest_ratio(image.height, image.width, JodiDataset.aspect_ratio)
    return str(ratio), height, width


def tab1():
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here", lines=3)
            negative_prompt = gr.Textbox(label="Negative Prompt", placeholder="Enter your negative prompt here")
            with gr.Group():
                with gr.Row():
                    num_inference_steps = gr.Slider(label="Number of Inference Steps", minimum=2, maximum=100, value=20, step=1)
                    guidance_scale = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=20.0, value=4.5, step=0.1)
                with gr.Row():
                    seed = gr.Slider(label="Seed", minimum=0, maximum=2147483647, value=1234, step=1)
                    batch_size = gr.Slider(label="Batch Size", minimum=1, maximum=20, value=1, step=1)
                with gr.Row():
                    ratio = gr.Dropdown(label="Aspect Ratio", choices=list(JodiDataset.aspect_ratio.keys()), value="1.0")
                    height = gr.Number(label="Height", interactive=False, value=1024, precision=0)
                    width = gr.Number(label="Width", interactive=False, value=1024, precision=0)
            generate_button = gr.Button("Generate")
        output_gallery = gr.Gallery(label="Generated Images", show_label=False, elem_id="gallery")

    ratio.input(change_height_width_by_ar, inputs=ratio, outputs=[height, width])

    def generate(prompt, negative_prompt, num_inference_steps, guidance_scale, height, width, seed, batch_size):  # noqa
        generator = torch.Generator(device=device).manual_seed(seed)
        outputs = pipe(
            images=[None] * (1 + pipe.num_conditions),
            role=[0] * (1 + pipe.num_conditions),
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=batch_size,
            generator=generator,
        )
        results = [post_processors[i](outputs[i]) for i in range(1 + pipe.num_conditions)]
        results = torch.stack(results, dim=1).reshape(-1, 3, height, width)
        results = [T.ToPILImage()(res).convert("RGB") for res in results.unbind(0)]
        return results

    generate_button.click(
        generate,
        inputs=[prompt, negative_prompt, num_inference_steps, guidance_scale, height, width, seed, batch_size],
        outputs=[output_gallery],
    )

    examples = [
        {
            "text": "A watercolor of a fox in a snowy field.",
            "ratio": "0.5", "height": 704, "width": 1408,
        },
        {
            "text": "A beachside resort with palm trees and a sunset.",
            "ratio": "0.67", "height": 768, "width": 1152,
        },
        {
            "text": "A pirate ship charges forward into a raging storm of thunder and lightning.",
            "ratio": "0.75", "height": 864, "width": 1152,
        },
        {
            "text": "A miniature castle stands in the center of the lake.",
            "ratio": "0.82", "height": 896, "width": 1088,
        },
        {
            "text": "A beautiful young woman with red hair and a red lipstick in a forest.",
            "ratio": "1.0", "height": 1024, "width": 1024,
        },
        {
            "text": "A beautiful sunset over the mountains with flowers growing on the ground.",
            "ratio": "1.21", "height": 1088, "width": 896,
        },
        {
            "text": "A lone tree on a hill under a starry night sky.",
            "ratio": "1.33", "height": 1152, "width": 864,
        },
        {
            "text": "A hot air balloon in the shape of a heart, Grand Canyon.",
            "ratio": "1.5", "height": 1152, "width": 768,
        },
        {
            "text": "A wizard with a long staff, looking out over a cliff edge.",
            "ratio": "2.0", "height": 1408, "width": 704,
        },
    ]

    gr.Examples(
        examples=[[e["text"], e["ratio"], e["height"], e["width"]] for e in examples],
        inputs=[prompt, ratio, height, width],
    )


def tab2():
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here", lines=3)
            negative_prompt = gr.Textbox(label="Negative Prompt", placeholder="Enter your negative prompt here")
            with gr.Group():
                with gr.Row():
                    num_inference_steps = gr.Slider(label="Number of Inference Steps", minimum=2, maximum=100, value=20, step=1)
                    guidance_scale = gr.Slider(label="Guidance Scale", minimum=1.0, maximum=20.0, value=4.5, step=0.1)
                with gr.Row():
                    seed = gr.Slider(label="Seed", minimum=0, maximum=2147483647, value=1234, step=1)
                    batch_size = gr.Slider(label="Batch Size", minimum=1, maximum=20, value=1, step=1)
                with gr.Row():
                    ratio = gr.Dropdown(label="Aspect Ratio", choices=list(JodiDataset.aspect_ratio.keys()), value="1.0")
                    height = gr.Number(label="Height", interactive=False, value=1024, precision=0)
                    width = gr.Number(label="Width", interactive=False, value=1024, precision=0)
            with gr.Group():
                control_images = []
                with gr.Row():
                    for label in pipe.config.conditions:  # type: ignore
                        control_images.append(gr.Image(label=label, type="pil", sources=["upload"]))
            generate_button = gr.Button("Generate")
        output_gallery = gr.Gallery(label="Generated Images", show_label=False, elem_id="gallery")

    ratio.input(change_height_width_by_ar, inputs=ratio, outputs=[height, width])

    def generate(prompt, negative_prompt, num_inference_steps, guidance_scale, height, width, seed, batch_size, *control_images):  # noqa
        role = [0] + [1 if control_image is not None else 2 for control_image in control_images]
        generator = torch.Generator(device=device).manual_seed(seed)
        outputs = pipe(
            images=[None] + list(control_images),
            role=role,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=batch_size,
            generator=generator,
        )
        results = post_processors[0](outputs[0])
        results = [T.ToPILImage()(res).convert("RGB") for res in results.unbind(0)]
        return results

    generate_button.click(
        generate,
        inputs=[prompt, negative_prompt, num_inference_steps, guidance_scale, height, width, seed, batch_size] + control_images,
        outputs=[output_gallery],
    )

    examples = [
        {
            "text": Path("./assets/test_images/1065-prompt.txt").read_text(encoding="utf-8").strip(),
            "control_images": [
                Image.open("./assets/test_images/1065-lineart.jpg").convert("RGB"),
                Image.open("./assets/test_images/1065-edge.jpg").convert("RGB"),
                Image.open("./assets/test_images/1065-depth.jpg").convert("RGB"),
                Image.open("./assets/test_images/1065-normal.jpg").convert("RGB"),
                None,
                None,
                None,
            ],
            "ratio": "0.67", "height": 768, "width": 1152,
        },
        {
            "text": Path("./assets/test_images/1-prompt.txt").read_text(encoding="utf-8").strip(),
            "control_images": [
                Image.open("./assets/test_images/1-lineart.jpg").convert("RGB"),
                Image.open("./assets/test_images/1-edge.jpg").convert("RGB"),
                Image.open("./assets/test_images/1-depth.jpg").convert("RGB"),
                Image.open("./assets/test_images/1-normal.jpg").convert("RGB"),
                Image.open("./assets/test_images/1-albedo.jpg").convert("RGB"),
                Image.open("./assets/test_images/1-seg.png").convert("RGB"),
                None,
            ],
            "ratio": "1.0", "height": 1024, "width": 1024,
        },
        {
            "text": Path("./assets/test_images/9-prompt.txt").read_text(encoding="utf-8").strip(),
            "control_images": [
                Image.open("./assets/test_images/9-lineart.jpg").convert("RGB"),
                Image.open("./assets/test_images/9-edge.jpg").convert("RGB"),
                Image.open("./assets/test_images/9-depth.jpg").convert("RGB"),
                Image.open("./assets/test_images/9-normal.jpg").convert("RGB"),
                Image.open("./assets/test_images/9-albedo.jpg").convert("RGB"),
                None,
                None,
            ],
            "ratio": "0.67", "height": 768, "width": 1152,
        },
        {
            "text": Path("./assets/test_images/5-prompt.txt").read_text(encoding="utf-8").strip(),
            "control_images": [
                None,
                None,
                Image.open("./assets/test_images/5-depth.png").convert("RGB"),
                Image.open("./assets/test_images/5-normal.png").convert("RGB"),
                Image.open("./assets/test_images/5-albedo.jpg").convert("RGB"),
                None,
                None,
            ],
            "ratio": "0.75", "height": 864, "width": 1152,
        },
        {
            "text": Path("./assets/test_images/7-prompt.txt").read_text(encoding="utf-8").strip(),
            "control_images": [
                Image.open("./assets/test_images/7-lineart.jpg").convert("RGB"),
                Image.open("./assets/test_images/7-edge.jpg").convert("RGB"),
                Image.open("./assets/test_images/7-depth.jpg").convert("RGB"),
                Image.open("./assets/test_images/7-normal.jpg").convert("RGB"),
                Image.open("./assets/test_images/7-albedo.jpg").convert("RGB"),
                Image.open("./assets/test_images/7-seg.png").convert("RGB"),
                Image.open("./assets/test_images/7-openpose.jpg").convert("RGB"),
            ],
            "ratio": "1.5", "height": 1152, "width": 768,
        },
    ]

    gr.Examples(
        examples=[[e["text"], *e["control_images"], e["ratio"], e["height"], e["width"]] for e in examples],
        inputs=[prompt, *control_images, ratio, height, width],
    )


def tab3():
    with gr.Row():
        with gr.Column():
            with gr.Group():
                with gr.Row():
                    seed = gr.Slider(label="Seed", minimum=0, maximum=2147483647, value=1234, step=1)
                    num_inference_steps = gr.Slider(label="Number of Inference Steps", minimum=2, maximum=100, value=10, step=1)
                with gr.Row():
                    ratio = gr.Dropdown(label="Aspect Ratio", choices=list(JodiDataset.aspect_ratio.keys()), value="1.0")
                    height = gr.Number(label="Height", interactive=False, value=1024, precision=0)
                    width = gr.Number(label="Width", interactive=False, value=1024, precision=0)
            with gr.Row():
                input_image = gr.Image(label=f"Input Image", type="pil", sources=["upload"])
                checkbox = []
                with gr.Column():
                    for label in pipe.config.conditions:  # type: ignore
                        checkbox.append(gr.Checkbox(label=label, value=True))
            generate_button = gr.Button("Generate")
        output_gallery = gr.Gallery(label="Generated Images", show_label=False, elem_id="gallery")

    input_image.upload(detect_aspect_ratio_from_image, inputs=input_image, outputs=[ratio, height, width])
    ratio.input(change_height_width_by_ar, inputs=ratio, outputs=[height, width])

    def generate(num_inference_steps, height, width, seed, input_image, *checkbox):  # noqa
        if all(not cb for cb in checkbox):
            raise gr.Error("Select at least one checkbox.")
        role = [1] + [0 if cb else 2 for cb in checkbox]
        generator = torch.Generator(device=device).manual_seed(seed)
        outputs = pipe(
            images=[input_image] + [None] * pipe.num_conditions,
            role=role,
            height=height,
            width=width,
            num_inference_steps=num_inference_steps,
            guidance_scale=1.0,
            num_images_per_prompt=1,
            generator=generator,
        )
        results = []
        for i in range(1 + pipe.num_conditions):
            if role[i] == 0:
                result = post_processors[i](outputs[i])
                results.append(result)
        results = torch.cat(results, dim=0).reshape(-1, 3, height, width)
        results = [T.ToPILImage()(res).convert("RGB") for res in results.unbind(0)]
        return results

    generate_button.click(
        generate,
        inputs=[num_inference_steps, height, width, seed, input_image] + checkbox,
        outputs=[output_gallery],
    )

    examples = [
        {
            "image": Image.open("./assets/test_images/pexels-pixabay-280221.jpg").convert("RGB"),
            "ratio": "0.67", "height": 768, "width": 1152,
        },
        {
            "image": Image.open("./assets/test_images/pexels-jplenio-1105378.jpg").convert("RGB"),
            "ratio": "0.67", "height": 768, "width": 1152,
        },
        {
            "image": Image.open("./assets/test_images/pexels-tobiasbjorkli-2335126.jpg").convert("RGB"),
            "ratio": "0.82", "height": 896, "width": 1088,
        },
        {
            "image": Image.open("./assets/test_images/pexels-jonathanborba-3255245.jpg").convert("RGB"),
            "ratio": "1.0", "height": 1024, "width": 1024,
        },
        {
            "image": Image.open("./assets/test_images/pexels-luis-dv-1453683203-31747630.jpg").convert("RGB"),
            "ratio": "1.21", "height": 1088, "width": 896,
        },
        {
            "image": Image.open("./assets/test_images/pexels-imagevain-2346018.jpg").convert("RGB"),
            "ratio": "1.21", "height": 1088, "width": 896,
        },
        {
            "image": Image.open("./assets/test_images/pexels-heyho-6538932.jpg").convert("RGB"),
            "ratio": "1.33", "height": 1152, "width": 864,
        },
        {
            "image": Image.open("./assets/test_images/pexels-laurathexplaura-3608263.jpg").convert("RGB"),
            "ratio": "1.5", "height": 1152, "width": 768,
        },
    ]

    gr.Examples(
        examples=[[e["image"], e["ratio"], e["height"], e["width"]] for e in examples],
        inputs=[input_image, ratio, height, width],
    )


if __name__ == "__main__":
    # Parse arguments
    args = get_parser().parse_args()

    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build pipeline
    pipe = JodiPipeline(args.config)
    pipe.from_pretrained(args.model_path)

    # Build post-processors
    post_processors: list[Any] = [ImagePostProcessor()]
    for condition in pipe.config.conditions:  # type: ignore
        if condition == "lineart":
            post_processors.append(LineartPostProcessor())
        elif condition == "edge":
            post_processors.append(EdgePostProcessor())
        elif condition == "depth":
            post_processors.append(DepthPostProcessor())
        elif condition == "normal":
            post_processors.append(NormalPostProcessor())
        elif condition == "albedo":
            post_processors.append(AlbedoPostProcessor())
        elif condition == "segmentation":
            post_processors.append(SegADE20KPostProcessor(color_scheme="colors12", only_return_image=True))
        elif condition == "openpose":
            post_processors.append(OpenposePostProcessor())
        else:
            print(f"Warning: Unknown condition: {condition}")
            post_processors.append(ImagePostProcessor())

    # Gradio UI
    blocks = gr.Blocks().queue()
    with blocks:
        with gr.Row():
            gr.Markdown("# Jodi")
        with gr.Tab(label="Joint Generation"):
            tab1()
        with gr.Tab(label="Controllable Generation"):
            tab2()
        with gr.Tab(label="Image Perception"):
            tab3()
    blocks.launch(server_name="0.0.0.0")
