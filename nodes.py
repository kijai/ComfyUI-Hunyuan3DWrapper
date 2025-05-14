import os
import torch
import torchvision.transforms as transforms
import torch.nn.functional as F
from PIL import Image
from pathlib import Path
import numpy as np
import json
import trimesh as Trimesh
from tqdm import tqdm

from .hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline, FaceReducer, FloaterRemover, DegenerateFaceRemover
from .hy3dgen.texgen.hunyuanpaint.unet.modules import UNet2DConditionModel, UNet2p5DConditionModel
from .hy3dgen.texgen.hunyuanpaint.pipeline import HunyuanPaintPipeline
from .hy3dgen.shapegen.schedulers import FlowMatchEulerDiscreteScheduler, ConsistencyFlowMatchEulerDiscreteScheduler
from .hy3dgen.shapegen.models.autoencoders import ShapeVAE

from diffusers import AutoencoderKL
from diffusers.schedulers import (
    DDIMScheduler, 
    PNDMScheduler, 
    DPMSolverMultistepScheduler, 
    EulerDiscreteScheduler, 
    EulerAncestralDiscreteScheduler,
    UniPCMultistepScheduler,
    HeunDiscreteScheduler,
    SASolverScheduler,
    DEISMultistepScheduler,
    LCMScheduler
    )

scheduler_mapping = {
    "DPM++": DPMSolverMultistepScheduler,
    "DPM++SDE": DPMSolverMultistepScheduler,
    "Euler": EulerDiscreteScheduler,
    "Euler A": EulerAncestralDiscreteScheduler,
    "PNDM": PNDMScheduler,
    "DDIM": DDIMScheduler,
    "SASolverScheduler": SASolverScheduler,
    "UniPCMultistepScheduler": UniPCMultistepScheduler,
    "HeunDiscreteScheduler": HeunDiscreteScheduler,
    "DEISMultistepScheduler": DEISMultistepScheduler,
    "LCMScheduler": LCMScheduler
}
available_schedulers = list(scheduler_mapping.keys())
from transformers import CLIPImageProcessor, CLIPTextModel, CLIPTokenizer

from accelerate import init_empty_weights
from accelerate.utils import set_module_tensor_to_device

import folder_paths

import comfy.model_management as mm
from comfy.utils import load_torch_file, ProgressBar, common_upscale

script_directory = os.path.dirname(os.path.abspath(__file__))

from .utils import log, print_memory

class ComfyProgressCallback:
    def __init__(self, total_steps):
        self.pbar = ProgressBar(total_steps)
        
    def __call__(self, pipe, i, t, callback_kwargs):
        self.pbar.update(1)
        return {
            "latents": callback_kwargs["latents"],
            "prompt_embeds": callback_kwargs["prompt_embeds"],
            "negative_prompt_embeds": callback_kwargs["negative_prompt_embeds"]
        }

class Hy3DTorchCompileSettings:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "backend": (["inductor","cudagraphs"], {"default": "inductor"}),
                "fullgraph": ("BOOLEAN", {"default": False, "tooltip": "Enable full graph mode"}),
                "mode": (["default", "max-autotune", "max-autotune-no-cudagraphs", "reduce-overhead"], {"default": "default"}),
                "dynamic": ("BOOLEAN", {"default": False, "tooltip": "Enable dynamic mode"}),
                "dynamo_cache_size_limit": ("INT", {"default": 64, "min": 0, "max": 1024, "step": 1, "tooltip": "torch._dynamo.config.cache_size_limit"}),
                "compile_transformer": ("BOOLEAN", {"default": True, "tooltip": "Compile single blocks"}),
                "compile_vae": ("BOOLEAN", {"default": True, "tooltip": "Compile double blocks"}),
            },
        }
    RETURN_TYPES = ("HY3DCOMPILEARGS",)
    RETURN_NAMES = ("torch_compile_args",)
    FUNCTION = "loadmodel"
    CATEGORY = "Hunyuan3DWrapper"
    DESCRIPTION = "torch.compile settings, when connected to the model loader, torch.compile of the selected layers is attempted. Requires Triton and torch 2.5.0 is recommended"

    def loadmodel(self, backend, fullgraph, mode, dynamic, dynamo_cache_size_limit, compile_transformer, compile_vae):

        compile_args = {
            "backend": backend,
            "fullgraph": fullgraph,
            "mode": mode,
            "dynamic": dynamic,
            "dynamo_cache_size_limit": dynamo_cache_size_limit,
            "compile_transformer": compile_transformer,
            "compile_vae": compile_vae,
        }

        return (compile_args, )
    
#region Model loading
class Hy3DModelLoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (folder_paths.get_filename_list("diffusion_models"), {"tooltip": "These models are loaded from the 'ComfyUI/models/diffusion_models' -folder",}),
            },
            "optional": {
                "compile_args": ("HY3DCOMPILEARGS", {"tooltip": "torch.compile settings, when connected to the model loader, torch.compile of the selected models is attempted. Requires Triton and torch 2.5.0 is recommended"}),
                "attention_mode": (["sdpa", "sageattn"], {"default": "sdpa"}),
                "cublas_ops": ("BOOLEAN", {"default": False, "tooltip": "Enable optimized cublas linear layers, speeds up decoding: https://github.com/aredden/torch-cublas-hgemm"}),
            }
        }

    RETURN_TYPES = ("HY3DMODEL", "HY3DVAE")
    RETURN_NAMES = ("pipeline", "vae")
    FUNCTION = "loadmodel"
    CATEGORY = "Hunyuan3DWrapper"

    def loadmodel(self, model, compile_args=None, attention_mode="sdpa", cublas_ops=False):
        device = mm.get_torch_device()
        offload_device=mm.unet_offload_device()

        model_path = folder_paths.get_full_path("diffusion_models", model)
        pipe, vae = Hunyuan3DDiTFlowMatchingPipeline.from_single_file(
            ckpt_path=model_path,  
            use_safetensors=True, 
            device=device, 
            offload_device=offload_device,
            compile_args=compile_args,
            attention_mode=attention_mode,
            cublas_ops=cublas_ops)
        
        return (pipe, vae,)
    
class Hy3DVAELoader:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("vae"), {"tooltip": "These models are loaded from 'ComfyUI/models/vae'"}),
            },
        }

    RETURN_TYPES = ("HY3DVAE",)
    RETURN_NAMES = ("vae",)
    FUNCTION = "loadmodel"
    CATEGORY = "Hunyuan3DWrapper"

    def loadmodel(self, model_name):
        device = mm.get_torch_device()
        offload_device=mm.unet_offload_device()

        model_path = folder_paths.get_full_path("vae", model_name)

        vae_sd = load_torch_file(model_path)
        geo_decoder_mlp_expand_ratio: 4
  
        mlp_expand_ratio = 4
        downsample_ratio = 1
        geo_decoder_ln_post = True
        if "geo_decoder.ln_post.weight" not in vae_sd:
            log.info("Turbo VAE detected")
            geo_decoder_ln_post = False
            mlp_expand_ratio = 1
            downsample_ratio = 2
            

        config = {
            'num_latents': 3072,
            'embed_dim': 64,
            'num_freqs': 8,
            'include_pi': False,
            'heads': 16,
            'width': 1024,
            'num_decoder_layers': 16,
            'qkv_bias': False,
            'qk_norm': True,
            'scale_factor': 0.9990943042622529,
            'geo_decoder_mlp_expand_ratio': mlp_expand_ratio,
            'geo_decoder_downsample_ratio': downsample_ratio,
            'geo_decoder_ln_post': geo_decoder_ln_post
        }

        vae = ShapeVAE(**config)
        vae.load_state_dict(vae_sd)
        vae.eval().to(torch.float16)
        
        return (vae,)

class DownloadAndLoadHy3DDelightModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (["hunyuan3d-delight-v2-0"],),
            },
            "optional": {
                "compile_args": ("HY3DCOMPILEARGS", {"tooltip": "torch.compile settings, when connected to the model loader, torch.compile of the selected models is attempted. Requires Triton and torch 2.5.0 is recommended"}),
            }
        }

    RETURN_TYPES = ("HY3DDIFFUSERSPIPE",)
    RETURN_NAMES = ("delight_pipe", )
    FUNCTION = "loadmodel"
    CATEGORY = "Hunyuan3DWrapper"

    def loadmodel(self, model, compile_args=None):
        device = mm.get_torch_device()

        download_path = os.path.join(folder_paths.models_dir,"diffusers")
        model_path = os.path.join(download_path, model)
        
        if not os.path.exists(model_path):
            log.info(f"Downloading model to: {model_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id="tencent/Hunyuan3D-2",
                allow_patterns=["*hunyuan3d-delight-v2-0*"],
                local_dir=download_path,
                local_dir_use_symlinks=False,
            )

        from diffusers import StableDiffusionInstructPix2PixPipeline, EulerAncestralDiscreteScheduler

        delight_pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
        delight_pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(delight_pipe.scheduler.config)
        delight_pipe = delight_pipe.to(device, torch.float16)

        

        if compile_args is not None:
            torch._dynamo.config.cache_size_limit = compile_args["dynamo_cache_size_limit"]
            if compile_args["compile_transformer"]:
                delight_pipe.unet = torch.compile(delight_pipe.unet)
            if compile_args["compile_vae"]:
                delight_pipe.vae = torch.compile(delight_pipe.vae)
        else:
            delight_pipe.enable_model_cpu_offload()
        
        return (delight_pipe,)
        
class Hy3DDelightImage:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "delight_pipe": ("HY3DDIFFUSERSPIPE",),
                "image": ("IMAGE", ),
                "steps": ("INT", {"default": 50, "min": 1}),
                "width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 16}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 16}),
                "cfg_image": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.01}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffffffffffff}),
        },
        "optional": {
            "scheduler": ("NOISESCHEDULER",),
        }
    }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process"
    CATEGORY = "Hunyuan3DWrapper"

    def process(self, delight_pipe, image, width, height, cfg_image, steps, seed, scheduler=None):

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()
        print("image in shape", image.shape)
        if scheduler is not None:
            if not hasattr(self, "default_scheduler"):
                self.default_scheduler = delight_pipe.scheduler
            delight_pipe.scheduler = scheduler
        else:
            if hasattr(self, "default_scheduler"):
                delight_pipe.scheduler = self.default_scheduler

        image = image.permute(0, 3, 1, 2).to(device)
        image = common_upscale(image, width, height, "lanczos", "disabled")
        

        images_list = []
        for img in image:
            out = delight_pipe(
                prompt="",
                image=img,
                generator=torch.manual_seed(seed),
                height=height,
                width=width,
                num_inference_steps=steps,
                image_guidance_scale=cfg_image,
                guidance_scale=1.0 if cfg_image == 1.0 else 1.01, #enable cfg for image, value doesn't matter as it do anything for text anyway
                output_type="pt",
                
            ).images[0]
            images_list.append(out)

        out_tensor = torch.stack(images_list).permute(0, 2, 3, 1).cpu().float()
        
        return (out_tensor, )
    
class DownloadAndLoadHy3DPaintModel:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (["hunyuan3d-paint-v2-0", "hunyuan3d-paint-v2-0-turbo"],),
            },
            "optional": {
                "compile_args": ("HY3DCOMPILEARGS", {"tooltip": "torch.compile settings, when connected to the model loader, torch.compile of the selected models is attempted. Requires Triton and torch 2.5.0 is recommended"}),
            }
        }

    RETURN_TYPES = ("HY3DDIFFUSERSPIPE",)
    RETURN_NAMES = ("multiview_pipe", )
    FUNCTION = "loadmodel"
    CATEGORY = "Hunyuan3DWrapper"

    def loadmodel(self, model, compile_args=None):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        download_path = os.path.join(folder_paths.models_dir,"diffusers")
        model_path = os.path.join(download_path, model)
        
        if not os.path.exists(model_path):
            log.info(f"Downloading model to: {model_path}")
            from huggingface_hub import snapshot_download
            snapshot_download(
                repo_id="tencent/Hunyuan3D-2",
                allow_patterns=[f"*{model}*"],
                ignore_patterns=["*unet/diffusion_pytorch_model.bin", "*image_encoder*"],
                local_dir=download_path,
                local_dir_use_symlinks=False,
            )

        torch_dtype = torch.float16
        config_path = os.path.join(model_path, 'unet', 'config.json')
        unet_ckpt_path_safetensors = os.path.join(model_path, 'unet','diffusion_pytorch_model.safetensors')
        unet_ckpt_path_bin = os.path.join(model_path, 'unet','diffusion_pytorch_model.bin')

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config not found at {config_path}")
        

        with open(config_path, 'r', encoding='utf-8') as file:
            config = json.load(file)

        with init_empty_weights():
            unet = UNet2DConditionModel(**config)
            unet = UNet2p5DConditionModel(unet)

        # Try loading safetensors first, fall back to .bin
        if os.path.exists(unet_ckpt_path_safetensors):
            import safetensors.torch
            unet_sd = safetensors.torch.load_file(unet_ckpt_path_safetensors)
        elif os.path.exists(unet_ckpt_path_bin):
            unet_sd = torch.load(unet_ckpt_path_bin, map_location='cpu', weights_only=True)
        else:
            raise FileNotFoundError(f"No checkpoint found at {unet_ckpt_path_safetensors} or {unet_ckpt_path_bin}")

        #unet.load_state_dict(unet_ckpt, strict=True)
        for name, param in unet.named_parameters():
            set_module_tensor_to_device(unet, name, device=offload_device, dtype=torch_dtype, value=unet_sd[name])

        vae = AutoencoderKL.from_pretrained(model_path, subfolder="vae", device=device, torch_dtype=torch_dtype)
        clip = CLIPTextModel.from_pretrained(model_path, subfolder="text_encoder", torch_dtype=torch_dtype)
        tokenizer = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        scheduler = EulerAncestralDiscreteScheduler.from_pretrained(model_path, subfolder="scheduler")
        feature_extractor = CLIPImageProcessor.from_pretrained(model_path, subfolder="feature_extractor")

        pipeline = HunyuanPaintPipeline(
            unet=unet,
            vae = vae,
            text_encoder=clip,
            tokenizer=tokenizer,
            scheduler=scheduler,
            feature_extractor=feature_extractor,
            )
        
        if compile_args is not None:
            pipeline.to(device)
            torch._dynamo.config.cache_size_limit = compile_args["dynamo_cache_size_limit"]
            if compile_args["compile_transformer"]:
                pipeline.unet = torch.compile(pipeline.unet)
            if compile_args["compile_vae"]:
                pipeline.vae = torch.compile(pipeline.vae)
        else:
            pipeline.enable_model_cpu_offload()
        return (pipeline,)

#region Texture
class Hy3DCameraConfig:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "camera_azimuths": ("STRING", {"default": "0, 90, 180, 270, 0, 180", "multiline": False}),
                "camera_elevations": ("STRING", {"default": "0, 0, 0, 0, 90, -90", "multiline": False}),
                "view_weights": ("STRING", {"default": "1, 0.1, 0.5, 0.1, 0.05, 0.05", "multiline": False}),
                "camera_distance": ("FLOAT", {"default": 1.45, "min": 0.1, "max": 10.0, "step": 0.001}),
                "ortho_scale": ("FLOAT", {"default": 1.2, "min": 0.1, "max": 10.0, "step": 0.001}),
            },
        }

    RETURN_TYPES = ("HY3DCAMERA",)
    RETURN_NAMES = ("camera_config",)
    FUNCTION = "process"
    CATEGORY = "Hunyuan3DWrapper"

    def process(self, camera_azimuths, camera_elevations, view_weights, camera_distance, ortho_scale):
        angles_list = list(map(int, camera_azimuths.replace(" ", "").split(',')))
        elevations_list = list(map(int, camera_elevations.replace(" ", "").split(',')))
        weights_list = list(map(float, view_weights.replace(" ", "").split(',')))

        camera_config = {
            "selected_camera_azims": angles_list,
            "selected_camera_elevs": elevations_list,
            "selected_view_weights": weights_list,
            "camera_distance": camera_distance,
            "ortho_scale": ortho_scale,
            }
        
        return (camera_config,)
    
class Hy3DMeshUVWrap:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
            },
        }

    RETURN_TYPES = ("TRIMESH", )
    RETURN_NAMES = ("trimesh", )
    FUNCTION = "process"
    CATEGORY = "Hunyuan3DWrapper"

    def process(self, trimesh):
        from .hy3dgen.texgen.utils.uv_warp_utils import mesh_uv_wrap
        trimesh = mesh_uv_wrap(trimesh)
        
        return (trimesh,)

class Hy3DRenderMultiView:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "render_size": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 16}),
                "texture_size": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 16}),
            },
            "optional": {
                "camera_config": ("HY3DCAMERA",),
                "normal_space": (["world", "tangent"], {"default": "world"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "IMAGE", "MESHRENDER", "MASK",)
    RETURN_NAMES = ("normal_maps", "position_maps", "renderer", "masks")
    FUNCTION = "process"
    CATEGORY = "Hunyuan3DWrapper"

    def process(self, trimesh, render_size, texture_size, camera_config=None, normal_space="world"):

        from .hy3dgen.texgen.differentiable_renderer.mesh_render import MeshRender

        if camera_config is None:
            selected_camera_azims = [0, 90, 180, 270, 0, 180]
            selected_camera_elevs = [0, 0, 0, 0, 90, -90]
            camera_distance = 1.45
            ortho_scale = 1.2
        else:
            selected_camera_azims = camera_config["selected_camera_azims"]
            selected_camera_elevs = camera_config["selected_camera_elevs"]
            camera_distance = camera_config["camera_distance"]
            ortho_scale = camera_config["ortho_scale"]
        
        self.render = MeshRender(
            default_resolution=render_size,
            texture_size=texture_size,
            camera_distance=camera_distance,
            ortho_scale=ortho_scale)

        self.render.load_mesh(trimesh)

        if normal_space == "world":
            normal_maps, masks = self.render_normal_multiview(
                selected_camera_elevs, selected_camera_azims, use_abs_coor=True)
            normal_tensors = torch.stack(normal_maps, dim=0)
            mask_tensors = torch.cat(masks, dim=0)
        elif normal_space == "tangent":
            normal_maps, masks = self.render_normal_multiview(
                selected_camera_elevs, selected_camera_azims, bg_color=[0, 0, 0], use_abs_coor=False)
            normal_tensors = torch.stack(normal_maps, dim=0)
            normal_tensors = 2.0 * normal_tensors - 1.0  # Map [0,1] to [-1,1]
            normal_tensors = normal_tensors / (torch.norm(normal_tensors, dim=-1, keepdim=True) + 1e-6)
            # Remap axes for standard normal map convention
            image = torch.zeros_like(normal_tensors)
            image[..., 0] = normal_tensors[..., 0]  # View right to R
            image[..., 1] = normal_tensors[..., 1]  # View up to G
            image[..., 2] = -normal_tensors[..., 2] # View forward (negated) to B

            # Create background color
            background_color = torch.tensor([0.502, 0.502, 1.0], device=normal_tensors.device) #8080FF

            mask_tensors = torch.cat(masks, dim=0)
            
            # Blend rendered image with background
            
            normal_tensors = (image + 1) * 0.5
            normal_tensors = normal_tensors * mask_tensors + background_color * (1 - mask_tensors)
            
        
        position_maps = self.render_position_multiview(
            selected_camera_elevs, selected_camera_azims)
        position_tensors = torch.stack(position_maps, dim=0)
        
        return (normal_tensors.cpu().float(), position_tensors.cpu().float(), self.render, mask_tensors.squeeze(-1).cpu().float(),)
    
    def render_normal_multiview(self, camera_elevs, camera_azims, use_abs_coor=True, bg_color=[1, 1, 1]):
        normal_maps = []
        masks = []
        for elev, azim in zip(camera_elevs, camera_azims):
            normal_map, mask = self.render.render_normal(
                elev, azim, bg_color=bg_color, use_abs_coor=use_abs_coor, return_type='th')
            normal_maps.append(normal_map)
            masks.append(mask)

        return normal_maps, masks

    def render_position_multiview(self, camera_elevs, camera_azims):
        position_maps = []
        for elev, azim in zip(camera_elevs, camera_azims):
            position_map = self.render.render_position(
                elev, azim, return_type='th')
            position_maps.append(position_map)

        return position_maps
    
class Hy3DRenderSingleView:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "render_type": (["normal", "depth"], {"default": "normal"}),
                "render_size": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 16}),
                "camera_type": (["orth", "perspective"], {"default": "orth"}),
                "camera_distance": ("FLOAT", {"default": 1.45, "min": 0.1, "max": 10.0, "step": 0.001}),
                "pan_x": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "pan_y": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "ortho_scale": ("FLOAT", {"default": 1.2, "min": 0.1, "max": 10.0, "step": 0.001}),
                "azimuth": ("FLOAT", {"default": 0, "min": -360, "max": 360, "step": 1}),
                "elevation": ("FLOAT", {"default": 0, "min": -360, "max": 360, "step": 1}),
                "bg_color": ("STRING", {"default": "128, 128, 255", "tooltip": "Color as RGB values in range 0-255, separated by commas."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask", )
    FUNCTION = "process"
    CATEGORY = "Hunyuan3DWrapper"

    def process(self, trimesh, render_type, camera_type, ortho_scale, camera_distance, pan_x, pan_y, render_size, azimuth, elevation, bg_color):

        from .hy3dgen.texgen.differentiable_renderer.mesh_render import MeshRender

        bg_color = [int(x.strip())/255.0 for x in bg_color.split(",")]

        self.render = MeshRender(
            default_resolution=render_size,
            texture_size=1024,
            camera_distance=camera_distance,
            camera_type=camera_type,
            ortho_scale=ortho_scale,
            filter_mode='linear'
            )

        self.render.load_mesh(trimesh)

        if render_type == "normal":
            normals, mask = self.render.render_normal(
                elevation,
                azimuth,
                camera_distance=camera_distance,
                center=None,
                resolution=render_size,
                bg_color=[0, 0, 0],
                use_abs_coor=False,
                pan_x=pan_x,
                pan_y=pan_y
            )

            normals = 2.0 * normals - 1.0  # Map [0,1] to [-1,1]
            normals = normals / (torch.norm(normals, dim=-1, keepdim=True) + 1e-6)
            # Remap axes for standard normal map convention
            image = torch.zeros_like(normals)
            image[..., 0] = normals[..., 0]  # View right to R
            image[..., 1] = normals[..., 1]  # View up to G
            image[..., 2] = -normals[..., 2] # View forward (negated) to B

            image = (image + 1) * 0.5

            #mask = mask.cpu().float()
            masked_image = image * mask

            bg_color = torch.tensor(bg_color, dtype=torch.float32, device=image.device)
            bg = bg_color.view(1, 1, 3) * (1.0 - mask)
            final_image = masked_image + bg
        elif render_type == "depth":
            depth, mask = self.render.render_depth(
                elevation,
                azimuth,
                camera_distance=camera_distance,
                center=None,
                resolution=render_size,
                pan_x=pan_x,
                pan_y=pan_y
            )
            final_image = depth.unsqueeze(0).repeat(1, 1, 1, 3)
        
        return (final_image.cpu().float(), mask.squeeze(-1).cpu().float(),)
    
class Hy3DRenderMultiViewDepth:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "render_size": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 16}),
                "texture_size": ("INT", {"default": 1024, "min": 64, "max": 4096, "step": 16}),
            },
            "optional": {
                "camera_config": ("HY3DCAMERA",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("depth_maps", "masks", )
    FUNCTION = "process"
    CATEGORY = "Hunyuan3DWrapper"

    def process(self, trimesh, render_size, texture_size, camera_config=None):

        mm.unload_all_models()
        mm.soft_empty_cache()

        from .hy3dgen.texgen.differentiable_renderer.mesh_render import MeshRender

        if camera_config is None:
            selected_camera_azims = [0, 90, 180, 270, 0, 180]
            selected_camera_elevs = [0, 0, 0, 0, 90, -90]
            camera_distance = 1.45
            ortho_scale = 1.2
        else:
            selected_camera_azims = camera_config["selected_camera_azims"]
            selected_camera_elevs = camera_config["selected_camera_elevs"]
            camera_distance = camera_config["camera_distance"]
            ortho_scale = camera_config["ortho_scale"]

        self.render = MeshRender(
            default_resolution=render_size,
            texture_size=texture_size,
            camera_distance=camera_distance,
            ortho_scale=ortho_scale)

        self.render.load_mesh(trimesh)

        depth_maps, masks = self.render_depth_multiview(
            selected_camera_elevs, selected_camera_azims)
        depth_tensors = torch.stack(depth_maps, dim=0)
        depth_tensors = depth_tensors.repeat(1, 1, 1, 3).cpu().float()
        masks = torch.cat(masks, dim=0).squeeze(-1).cpu().float()
        
        return (depth_tensors, masks,)
    
    def render_depth_multiview(self, camera_elevs, camera_azims):
        depth_maps = []
        masks = []
        for elev, azim in zip(camera_elevs, camera_azims):        
            depth_map, mask = self.render.render_depth(elev, azim, return_type='th')
            depth_maps.append(depth_map)
            masks.append(mask)

        return depth_maps, masks

class Hy3DDiffusersSchedulerConfig:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("HY3DDIFFUSERSPIPE",),
                "scheduler": (available_schedulers,
                    {
                        "default": 'Euler A'
                    }),
                "sigmas": (["default", "karras", "exponential", "beta"],),
            },
        }

    RETURN_TYPES = ("NOISESCHEDULER",)
    RETURN_NAMES = ("diffusers_scheduler",)
    FUNCTION = "process"
    CATEGORY = "Hunyuan3DWrapper"

    def process(self, pipeline, scheduler, sigmas):

        scheduler_config = dict(pipeline.scheduler.config)
        
        if scheduler in scheduler_mapping:
            if scheduler == "DPM++SDE":
                scheduler_config["algorithm_type"] = "sde-dpmsolver++"
            else:
                scheduler_config.pop("algorithm_type", None)
            if sigmas == "default":
                scheduler_config["use_karras_sigmas"] = False
                scheduler_config["use_exponential_sigmas"] = False
                scheduler_config["use_beta_sigmas"] = False
            elif sigmas == "karras":
                scheduler_config["use_karras_sigmas"] = True
                scheduler_config["use_exponential_sigmas"] = False
                scheduler_config["use_beta_sigmas"] = False
            elif sigmas == "exponential":
                scheduler_config["use_karras_sigmas"] = False
                scheduler_config["use_exponential_sigmas"] = True
                scheduler_config["use_beta_sigmas"] = False
            elif sigmas == "beta":
                scheduler_config["use_karras_sigmas"] = False
                scheduler_config["use_exponential_sigmas"] = False
                scheduler_config["use_beta_sigmas"] = True
            noise_scheduler = scheduler_mapping[scheduler].from_config(scheduler_config)
        else:
            raise ValueError(f"Unknown scheduler: {scheduler}")
        
        return (noise_scheduler,)
    
class Hy3DSampleMultiView:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("HY3DDIFFUSERSPIPE",),
                "ref_image": ("IMAGE", ),
                "normal_maps": ("IMAGE", ),
                "position_maps": ("IMAGE", ),
                "view_size": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 16}),
                "steps": ("INT", {"default": 30, "min": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "camera_config": ("HY3DCAMERA",),
                "scheduler": ("NOISESCHEDULER",),
                "denoise_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "samples": ("LATENT", ),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process"
    CATEGORY = "Hunyuan3DWrapper"

    def process(self, pipeline, ref_image, normal_maps, position_maps, view_size, seed, steps, 
                camera_config=None, scheduler=None, denoise_strength=1.0, samples=None):
        device = mm.get_torch_device()
        mm.unload_all_models()
        mm.soft_empty_cache()
        torch.manual_seed(seed)
        generator=torch.Generator(device=pipeline.device).manual_seed(seed)

        input_image = ref_image.permute(0, 3, 1, 2).unsqueeze(0).to(device)

        device = mm.get_torch_device()

        if camera_config is None:
            selected_camera_azims = [0, 90, 180, 270, 0, 180]
            selected_camera_elevs = [0, 0, 0, 0, 90, -90]
        else:
            selected_camera_azims = camera_config["selected_camera_azims"]
            selected_camera_elevs = camera_config["selected_camera_elevs"]
        
        camera_info = [(((azim // 30) + 9) % 12) // {-90: 3, -45: 2, -20: 1, 0: 1, 20: 1, 45: 2, 90: 3}[
            elev] + {-90: 36, -45: 30, -20: 0, 0: 12, 20: 24, 45: 30, 90: 40}[elev] for azim, elev in
                    zip(selected_camera_azims, selected_camera_elevs)]
        #print(camera_info)
        
        normal_maps_np = (normal_maps * 255).to(torch.uint8).cpu().numpy()
        normal_maps_pil = [Image.fromarray(normal_map) for normal_map in normal_maps_np]

        position_maps_np = (position_maps * 255).to(torch.uint8).cpu().numpy()
        position_maps_pil = [Image.fromarray(position_map) for position_map in position_maps_np]
        
        control_images = normal_maps_pil + position_maps_pil

        for i in range(len(control_images)):
            control_images[i] = control_images[i].resize((view_size, view_size))
            if control_images[i].mode == 'L':
                control_images[i] = control_images[i].point(lambda x: 255 if x > 1 else 0, mode='1')

        num_view = len(control_images) // 2
        normal_image = [[control_images[i] for i in range(num_view)]]
        position_image = [[control_images[i + num_view] for i in range(num_view)]]

        callback = ComfyProgressCallback(total_steps=steps)

        if scheduler is not None:
            if not hasattr(self, "default_scheduler"):
                self.default_scheduler = pipeline.scheduler
            pipeline.scheduler = scheduler
        else:
            if hasattr(self, "default_scheduler"):
                pipeline.scheduler = self.default_scheduler

        multiview_images = pipeline(
            input_image,
            width=view_size,
            height=view_size,
            generator=generator,
            latents=samples["samples"] if samples is not None else None,
            num_in_batch = num_view,
            camera_info_gen = [camera_info],
            camera_info_ref = [[0]],
            normal_imgs = normal_image,
            position_imgs = position_image,
            num_inference_steps=steps,
            output_type="pt",
            callback_on_step_end=callback,
            callback_on_step_end_tensor_inputs=["latents", "prompt_embeds", "negative_prompt_embeds"],
            denoise_strength=denoise_strength
            ).images

        out_tensors = multiview_images.permute(0, 2, 3, 1).cpu().float()
        
        return (out_tensors,)
    
class Hy3DBakeFromMultiview:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
                "renderer": ("MESHRENDER",),
            },
            "optional": {
                "camera_config": ("HY3DCAMERA",),
            }
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MESHRENDER") 
    RETURN_NAMES = ("texture", "mask", "renderer")
    FUNCTION = "process"
    CATEGORY = "Hunyuan3DWrapper"

    def process(self, images, renderer, camera_config=None):
        device = mm.get_torch_device()
        self.render = renderer

        multiviews = images.permute(0, 3, 1, 2)
        multiviews = multiviews.cpu().numpy()
        multiviews_pil = [Image.fromarray((image.transpose(1, 2, 0) * 255).astype(np.uint8)) for image in multiviews]

        if camera_config is None:
            selected_camera_azims = [0, 90, 180, 270, 0, 180]
            selected_camera_elevs = [0, 0, 0, 0, 90, -90]
            selected_view_weights = [1, 0.1, 0.5, 0.1, 0.05, 0.05]
        else:
            selected_camera_azims = camera_config["selected_camera_azims"]
            selected_camera_elevs = camera_config["selected_camera_elevs"]
            selected_view_weights = camera_config["selected_view_weights"]

        merge_method = 'fast'
        self.bake_exp = 4
        
        texture, mask = self.bake_from_multiview(multiviews_pil,
                                                 selected_camera_elevs, selected_camera_azims, selected_view_weights,
                                                 method=merge_method)
        
        
        mask = mask.squeeze(-1).cpu().float()
        texture = texture.unsqueeze(0).cpu().float()

        return (texture, mask, self.render)
    
    def bake_from_multiview(self, views, camera_elevs,
                            camera_azims, view_weights, method='graphcut'):
        project_textures, project_weighted_cos_maps = [], []
        project_boundary_maps = []
        pbar = ProgressBar(len(views))
        for view, camera_elev, camera_azim, weight in zip(
            views, camera_elevs, camera_azims, view_weights):
            project_texture, project_cos_map, project_boundary_map = self.render.back_project(
                view, camera_elev, camera_azim)
            project_cos_map = weight * (project_cos_map ** self.bake_exp)
            project_textures.append(project_texture)
            project_weighted_cos_maps.append(project_cos_map)
            project_boundary_maps.append(project_boundary_map)
            pbar.update(1)

        if method == 'fast':
            texture, ori_trust_map = self.render.fast_bake_texture(
                project_textures, project_weighted_cos_maps)
        else:
            raise f'no method {method}'
        return texture, ori_trust_map > 1E-8
    
class Hy3DMeshVerticeInpaintTexture:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "texture": ("IMAGE", ),
                "mask": ("MASK", ),
                "renderer": ("MESHRENDER",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MESHRENDER" ) 
    RETURN_NAMES = ("texture", "mask", "renderer" )
    FUNCTION = "process"
    CATEGORY = "Hunyuan3DWrapper"

    def process(self, texture, renderer, mask):
        from .hy3dgen.texgen.differentiable_renderer.mesh_processor import meshVerticeInpaint
        vtx_pos, pos_idx, vtx_uv, uv_idx = renderer.get_mesh()

        mask_np = (mask.squeeze(-1).squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        texture_np = texture.squeeze(0).cpu().numpy() * 255

        texture_np, mask_np = meshVerticeInpaint(
            texture_np, mask_np, vtx_pos, vtx_uv, pos_idx, uv_idx)
            
        texture_tensor = torch.from_numpy(texture_np).float() / 255.0
        texture_tensor = texture_tensor.unsqueeze(0)

        mask_tensor = torch.from_numpy(mask_np).float() / 255.0
        mask_tensor = mask_tensor.unsqueeze(0)
        
        return (texture_tensor, mask_tensor, renderer)

class CV2InpaintTexture:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "texture": ("IMAGE", ),
                "mask": ("MASK", ),
                "inpaint_radius": ("INT", {"default": 3, "min": 1, "max": 10, "step": 1}),
                "inpaint_method": (["ns", "telea"], {"default": "ns"}),
            },
        }

    RETURN_TYPES = ("IMAGE", ) 
    RETURN_NAMES = ("texture", )
    FUNCTION = "inpaint"
    CATEGORY = "Hunyuan3DWrapper"

    def inpaint(self, texture, mask, inpaint_radius, inpaint_method):
        import cv2
        mask = 1 - mask
        mask_np = (mask.squeeze(-1).squeeze(0).cpu().numpy() * 255).astype(np.uint8)
        texture_np = (texture.squeeze(0).cpu().numpy() * 255).astype(np.uint8)

        if inpaint_method == "ns":
            inpaint_algo = cv2.INPAINT_NS
        elif inpaint_method == "telea":
            inpaint_algo = cv2.INPAINT_TELEA
            
        texture_np = cv2.inpaint(
            texture_np,
            mask_np,
            inpaint_radius,
            inpaint_algo)
        
        texture_tensor = torch.from_numpy(texture_np).float() / 255.0
        texture_tensor = texture_tensor.unsqueeze(0)
        
        return (texture_tensor, )
    
class Hy3DApplyTexture:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "texture": ("IMAGE", ),
                "renderer": ("MESHRENDER",),
            },
        }

    RETURN_TYPES = ("TRIMESH", ) 
    RETURN_NAMES = ("trimesh", )
    FUNCTION = "apply"
    CATEGORY = "Hunyuan3DWrapper"

    def apply(self, texture, renderer):
        self.render = renderer
        self.render.set_texture(texture.squeeze(0))
        textured_mesh = self.render.save_mesh()
        
        return (textured_mesh,)

#region Mesh

class Hy3DLoadMesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "glb_path": ("STRING", {"default": "", "tooltip": "The glb path with mesh to load."}), 
            }
        }
    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    OUTPUT_TOOLTIPS = ("The glb model with mesh to texturize.",)
    
    FUNCTION = "load"
    CATEGORY = "Hunyuan3DWrapper"
    DESCRIPTION = "Loads a glb model from the given path."

    def load(self, glb_path):

        if not os.path.exists(glb_path):
            glb_path = os.path.join(folder_paths.get_input_directory(), glb_path)
        
        trimesh = Trimesh.load(glb_path, force="mesh")
        
        return (trimesh,)


class TrimeshToMESH:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
            }
        }
    RETURN_TYPES = ("MESH",)
    OUTPUT_TOOLTIPS = ("MESH object containing vertices and faces as torch tensors.",)
    
    FUNCTION = "load"
    CATEGORY = "Hunyuan3DWrapper"
    DESCRIPTION = "Converts trimesh object to ComfyUI MESH object, which only includes mesh data"

    def load(self, trimesh):

        vertices = torch.tensor(trimesh.vertices, dtype=torch.float32)
        faces = torch.tensor(trimesh.faces, dtype=torch.float32)
        mesh = (self.MESH(vertices.unsqueeze(0), faces.unsqueeze(0)))
        
        return (mesh,)

    class MESH:
        def __init__(self, vertices, faces):
            self.vertices = vertices
            self.faces = faces

class MESHToTrimesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mesh": ("MESH",),
            }
        }
    RETURN_TYPES = ("TRIMESH",)
    OUTPUT_TOOLTIPS = ("TRIMESH object containing vertices and faces as torch tensors.",)
    
    FUNCTION = "load"
    CATEGORY = "Hunyuan3DWrapper"
    DESCRIPTION = "Converts trimesh object to ComfyUI MESH object, which only includes mesh data"

    def load(self, mesh):
        mesh_output = Trimesh.Trimesh(mesh.vertices[0], mesh.faces[0])
        return (mesh_output,)

class Hy3DUploadMesh:
    @classmethod
    def INPUT_TYPES(s):
        mesh_extensions = ['glb', 'gltf', 'obj', 'ply', 'stl', '3mf']
        input_dir = folder_paths.get_input_directory()
        files = []
        for f in os.listdir(input_dir):
            if os.path.isfile(os.path.join(input_dir, f)):
                file_parts = f.split('.')
                if len(file_parts) > 1 and (file_parts[-1] in mesh_extensions):
                    files.append(f)
        return {
            "required": {
                "mesh": (sorted(files),),
            }
        }
    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    OUTPUT_TOOLTIPS = ("The glb model with mesh to texturize.",)
    
    FUNCTION = "load"
    CATEGORY = "Hunyuan3DWrapper"
    DESCRIPTION = "Loads a glb model from the given path."

    def load(self, mesh):
        path = mesh.strip()
        if path.startswith("\""):
            path = path[1:]
        if path.endswith("\""):
            path = path[:-1]
        mesh_file = folder_paths.get_annotated_filepath(path)
        loaded_mesh = Trimesh.load(mesh_file, force="mesh")
        
        return (loaded_mesh,)


class Hy3DGenerateMesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("HY3DMODEL",),
                "image": ("IMAGE", ),
                "guidance_scale": ("FLOAT", {"default": 5.5, "min": 0.0, "max": 100.0, "step": 0.01}),
                "steps": ("INT", {"default": 30, "min": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "mask": ("MASK", ),
                "scheduler": (["FlowMatchEulerDiscreteScheduler", "ConsistencyFlowMatchEulerDiscreteScheduler"],),
                "force_offload": ("BOOLEAN", {"default": True, "tooltip": "Offloads the model to the offload device once the process is done."}),
            }
        }

    RETURN_TYPES = ("HY3DLATENT",)
    RETURN_NAMES = ("latents",)
    FUNCTION = "process"
    CATEGORY = "Hunyuan3DWrapper"

    def process(self, pipeline, image, steps, guidance_scale, seed, mask=None, front=None, back=None, left=None, right=None, 
                scheduler="FlowMatchEulerDiscreteScheduler", force_offload=True):

        mm.unload_all_models()
        mm.soft_empty_cache()

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        image = image.permute(0, 3, 1, 2).to(device)
        image = image * 2 - 1

        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, 3, 1, 1).to(device)
            if mask.shape[2] != image.shape[2] or mask.shape[3] != image.shape[3]:
                mask = F.interpolate(mask, size=(image.shape[2], image.shape[3]), mode='nearest')

        if scheduler == "FlowMatchEulerDiscreteScheduler":
            scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
        elif scheduler == "ConsistencyFlowMatchEulerDiscreteScheduler":
            scheduler = ConsistencyFlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, pcm_timesteps=100)

        pipeline.scheduler = scheduler

        pipeline.to(device)

        try:
            torch.cuda.reset_peak_memory_stats(device)
        except:
            pass

        latents = pipeline(
            image=image, 
            mask=mask,
            num_inference_steps=steps, 
            guidance_scale=guidance_scale,
            generator=torch.manual_seed(seed))

        print_memory(device)
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except:
            pass
        
        if not force_offload:
            pipeline.to(offload_device)
        
        return (latents, )
    
class Hy3DGenerateMeshMultiView():
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pipeline": ("HY3DMODEL",),
                "guidance_scale": ("FLOAT", {"default": 5.5, "min": 0.0, "max": 100.0, "step": 0.01}),
                "steps": ("INT", {"default": 30, "min": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "front": ("IMAGE", ),
                "left": ("IMAGE", ),
                "right": ("IMAGE", ),
                "back": ("IMAGE", ),
                "scheduler": (["FlowMatchEulerDiscreteScheduler", "ConsistencyFlowMatchEulerDiscreteScheduler"],),           
            }
        }

    RETURN_TYPES = ("HY3DLATENT", "IMAGE", "MASK",)
    RETURN_NAMES = ("latents", "image", "mask")
    FUNCTION = "process"
    CATEGORY = "Hunyuan3DWrapper"

    def process(self, pipeline, steps, guidance_scale, seed, mask=None, front=None, back=None, left=None, right=None, scheduler="FlowMatchEulerDiscreteScheduler"):

        mm.unload_all_models()
        mm.soft_empty_cache()

        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        pipeline.to(device)

        if front is not None and not torch.all(front < 1e-6).item():
            front = front.clone().permute(0, 3, 1, 2).to(device)
        else:
            front = None
        if back is not None and not torch.all(back < 1e-6).item():
            back = back.clone().permute(0, 3, 1, 2).to(device)
        else:
            back = None
        if left is not None and not torch.all(left < 1e-6).item():
            left = left.clone().permute(0, 3, 1, 2).to(device)
        else:
            left = None
        if right is not None and not torch.all(right < 1e-6).item():
            right = right.clone().permute(0, 3, 1, 2).to(device)
        else:
            right = None
            
        view_dict = {
            'front': front,
            'left': left,
            'right': right,
            'back': back
        }

        if scheduler == "FlowMatchEulerDiscreteScheduler":
            scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000)
        elif scheduler == "ConsistencyFlowMatchEulerDiscreteScheduler":
            scheduler = ConsistencyFlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, pcm_timesteps=100)

        try:
            torch.cuda.reset_peak_memory_stats(device)
        except:
            pass

        latents = pipeline(
            image=None, 
            mask=mask,
            num_inference_steps=steps, 
            guidance_scale=guidance_scale,
            generator=torch.manual_seed(seed),
            view_dict=view_dict)

        print_memory(device)
        try:
            torch.cuda.reset_peak_memory_stats(device)
        except:
            pass
        
        images = []
        masks = []
        for view_tag, view_image in view_dict.items():
            if view_image is not None:
                if view_image.shape[1] == 4:
                    rgb = view_image[:, :3, :, :]
                    alpha = view_image[:, 3:4, :, :]
                    mask = alpha
                    masks.append(mask)
                else:
                    rgb = view_image
                images.append(rgb)

        image_tensors = torch.cat(images, 0).permute(0, 2, 3, 1).cpu().float()
        if masks:
            mask_tensors = torch.cat(masks, 0).squeeze(1).cpu().float()
        else:
            mask_tensors = torch.zeros(image_tensors.shape[0], image_tensors.shape[1], image_tensors.shape[2]).cpu().float()

        pipeline.to(offload_device)
        
        return (latents, image_tensors, mask_tensors)
    
class Hy3DVAEDecode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": ("HY3DVAE",),
                "latents": ("HY3DLATENT", ),
                "box_v": ("FLOAT", {"default": 1.01, "min": -10.0, "max": 10.0, "step": 0.001}),
                "octree_resolution": ("INT", {"default": 384, "min": 8, "max": 4096, "step": 8}),
                "num_chunks": ("INT", {"default": 8000, "min": 1, "max": 10000000, "step": 1, "tooltip": "Number of chunks to process at once, higher values use more memory, but make the process faster"}),
                "mc_level": ("FLOAT", {"default": 0, "min": -1.0, "max": 1.0, "step": 0.0001}),
                #"mc_algo": (["mc", "dmc", "odc", "none"], {"default": "mc"}),
                "mc_algo": (["mc", "dmc"], {"default": "mc"}),
            },
            "optional": {
                "enable_flash_vdm": ("BOOLEAN", {"default": True}),
                "force_offload": ("BOOLEAN", {"default": True, "tooltip": "Offloads the model to the offload device once the process is done."}),

            }
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    FUNCTION = "process"
    CATEGORY = "Hunyuan3DWrapper"

    def process(self, vae, latents, box_v, octree_resolution, mc_level, num_chunks, mc_algo, enable_flash_vdm=True, force_offload=True):
        device = mm.get_torch_device()
        offload_device = mm.unet_offload_device()

        vae.to(device)

        vae.enable_flashvdm_decoder(
            enabled=enable_flash_vdm,
            mc_algo=mc_algo,)
        
        latents = 1. / vae.scale_factor * latents
        latents = vae(latents)
        
        outputs = vae.latents2mesh(
            latents,
            bounds=box_v,
            mc_level=mc_level,
            num_chunks=num_chunks,
            octree_resolution=octree_resolution,
            mc_algo=mc_algo,
        )[0]
        if force_offload:
            vae.to(offload_device)

        outputs.mesh_f = outputs.mesh_f[:, ::-1]
        mesh_output = Trimesh.Trimesh(outputs.mesh_v, outputs.mesh_f)
        log.info(f"Decoded mesh with {mesh_output.vertices.shape[0]} vertices and {mesh_output.faces.shape[0]} faces")
        
        return (mesh_output, )

class Hy3DPostprocessMesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "remove_floaters": ("BOOLEAN", {"default": True}),
                "remove_degenerate_faces": ("BOOLEAN", {"default": True}),
                "reduce_faces": ("BOOLEAN", {"default": True}),
                "max_facenum": ("INT", {"default": 40000, "min": 1, "max": 10000000, "step": 1}),
                "smooth_normals": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    FUNCTION = "process"
    CATEGORY = "Hunyuan3DWrapper"

    def process(self, trimesh, remove_floaters, remove_degenerate_faces, reduce_faces, max_facenum, smooth_normals):
        new_mesh = trimesh.copy()
        if remove_floaters:
            new_mesh = FloaterRemover()(new_mesh)
            log.info(f"Removed floaters, resulting in {new_mesh.vertices.shape[0]} vertices and {new_mesh.faces.shape[0]} faces")
        if remove_degenerate_faces:
            new_mesh = DegenerateFaceRemover()(new_mesh)
            log.info(f"Removed degenerate faces, resulting in {new_mesh.vertices.shape[0]} vertices and {new_mesh.faces.shape[0]} faces")
        if reduce_faces:
            new_mesh = FaceReducer()(new_mesh, max_facenum=max_facenum)
            log.info(f"Reduced faces, resulting in {new_mesh.vertices.shape[0]} vertices and {new_mesh.faces.shape[0]} faces")
        if smooth_normals:              
            new_mesh.vertex_normals = Trimesh.smoothing.get_vertices_normals(new_mesh)

        
        return (new_mesh, )

class Hy3DFastSimplifyMesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "target_count": ("INT", {"default": 40000, "min": 1, "max": 100000000, "step": 1, "tooltip": "Target number of triangles"}),
                "aggressiveness": ("INT", {"default": 7, "min": 0, "max": 100, "step": 1, "tooltip": "Parameter controlling the growth rate of the threshold at each iteration when lossless is False."}),
                "max_iterations": ("INT", {"default": 100, "min": 1, "max": 1000, "step": 1, "tooltip": "Maximal number of iterations"}),
                "update_rate": ("INT", {"default": 5, "min": 1, "max": 1000, "step": 1, "tooltip": "Number of iterations between each update"}),
                "preserve_border": ("BOOLEAN", {"default": True, "tooltip": "Flag for preserving the vertices situated on open borders."}),
                "lossless": ("BOOLEAN", {"default": False, "tooltip": "Flag for using the lossless simplification method. Sets the update rate to 1"}),
                "threshold_lossless": ("FLOAT", {"default": 1e-3, "min": 0.0, "max": 1.0, "step": 0.0001, "tooltip": "Threshold for the lossless simplification method."}),
            },
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    FUNCTION = "process"
    CATEGORY = "Hunyuan3DWrapper"
    DESCRIPTION = "Simplifies the mesh using Fast Quadric Mesh Reduction: https://github.com/Kramer84/pyfqmr-Fast-Quadric-Mesh-Reduction"

    def process(self, trimesh, target_count, aggressiveness, preserve_border, max_iterations,lossless, threshold_lossless, update_rate):
        new_mesh = trimesh.copy()
        try:
            import pyfqmr
        except ImportError:
            raise ImportError("pyfqmr not found. Please install it using 'pip install pyfqmr' https://github.com/Kramer84/pyfqmr-Fast-Quadric-Mesh-Reduction")
        
        mesh_simplifier = pyfqmr.Simplify()
        mesh_simplifier.setMesh(trimesh.vertices, trimesh.faces)
        mesh_simplifier.simplify_mesh(
            target_count=target_count, 
            aggressiveness=aggressiveness,
            update_rate=update_rate,
            max_iterations=max_iterations,
            preserve_border=preserve_border, 
            verbose=True,
            lossless=lossless,
            threshold_lossless=threshold_lossless
            )
        new_mesh.vertices, new_mesh.faces, _ = mesh_simplifier.getMesh()
        log.info(f"Simplified mesh to {target_count} vertices, resulting in {new_mesh.vertices.shape[0]} vertices and {new_mesh.faces.shape[0]} faces")   
        
        return (new_mesh, )
    
class Hy3DMeshInfo:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
            },
        }

    RETURN_TYPES = ("TRIMESH", "INT", "INT", )
    RETURN_NAMES = ("trimesh", "vertices", "faces",)
    FUNCTION = "process"
    CATEGORY = "Hunyuan3DWrapper"

    def process(self, trimesh):
        vertices_count = trimesh.vertices.shape[0]
        faces_count = trimesh.faces.shape[0]
        log.info(f"Hy3DMeshInfo: Mesh has {vertices_count} vertices and {trimesh.faces.shape[0]} faces")
        return {"ui": {
            "text": [f"{vertices_count:,.0f}x{faces_count:,.0f}"]}, 
            "result": (trimesh, vertices_count, faces_count) 
        }
    
class Hy3DIMRemesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "merge_vertices": ("BOOLEAN", {"default": True}),
                "vertex_count": ("INT", {"default": 10000, "min": 100, "max": 10000000, "step": 1}),
                "smooth_iter": ("INT", {"default": 8, "min": 0, "max": 100, "step": 1}),
                "align_to_boundaries": ("BOOLEAN", {"default": True}),
                "triangulate_result": ("BOOLEAN", {"default": True}),
                "max_facenum": ("INT", {"default": 40000, "min": 1, "max": 10000000, "step": 1}),
            },
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    FUNCTION = "remesh"
    CATEGORY = "Hunyuan3DWrapper"
    DESCRIPTION = "Remeshes the mesh using instant-meshes: https://github.com/wjakob/instant-meshes, Note: this will remove all vertex colors and textures."

    def remesh(self, trimesh, merge_vertices, vertex_count, smooth_iter, align_to_boundaries, triangulate_result, max_facenum):
        try:
            import pynanoinstantmeshes as PyNIM
        except ImportError:
            raise ImportError("pynanoinstantmeshes not found. Please install it using 'pip install pynanoinstantmeshes'")
        new_mesh = trimesh.copy()
        if merge_vertices:
            trimesh.merge_vertices(new_mesh)

        new_verts, new_faces = PyNIM.remesh(
            np.array(trimesh.vertices, dtype=np.float32),
            np.array(trimesh.faces, dtype=np.uint32),
            vertex_count,
            align_to_boundaries=align_to_boundaries,
            smooth_iter=smooth_iter
        )
        if new_verts.shape[0] - 1 != new_faces.max():
            # Skip test as the meshing failed
            raise ValueError("Instant-meshes failed to remesh the mesh")
        new_verts = new_verts.astype(np.float32)
        if triangulate_result:
            new_faces = Trimesh.geometry.triangulate_quads(new_faces)
        
        if len(new_mesh.faces) > max_facenum:
            new_mesh = FaceReducer()(new_mesh, max_facenum=max_facenum)

        return (new_mesh, )
    
class Hy3DBPT:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "enable_bpt": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 0xffffffff}),
                "temperature": ("FLOAT", {"default": 0.5}),
                "pc_num": ("INT", {"default": 4096, "min": 1024, "max": 8192, "step": 1024}),
                "samples": ("INT", {"default": 100000})
            },
        }

    RETURN_TYPES = ("TRIMESH",)
    RETURN_NAMES = ("trimesh",)
    FUNCTION = "bpt"
    CATEGORY = "Hunyuan3DWrapper"
    DESCRIPTION = "BPT the mesh using bpt: https://github.com/whaohan/bpt"
    
    def bpt(self, trimesh, enable_bpt, temperature, pc_num, seed, samples):
        mm.unload_all_models()
        mm.soft_empty_cache()
        new_mesh = trimesh.copy()
        if enable_bpt:
            from .hy3dgen.shapegen.postprocessors import BptMesh
            new_mesh = BptMesh()(new_mesh, with_normal=True, temperature=temperature, batch_size=1, pc_num=pc_num, verbose=False, seed=seed, samples=samples)
            mm.unload_all_models()
            mm.soft_empty_cache()

        return (new_mesh, )
            
class Hy3DGetMeshPBRTextures:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "texture" : (["base_color", "emissive", "metallic_roughness", "normal", "occlusion"], ),
            },
        }

    RETURN_TYPES = ("IMAGE", )
    RETURN_NAMES = ("image",)
    FUNCTION = "get_textures"
    CATEGORY = "Hunyuan3DWrapper"

    def get_textures(self, trimesh, texture):
        
        TEXTURE_MAPPING = {
            'base_color': ('baseColorTexture', "Base color"),
            'emissive': ('emissiveTexture', "Emissive"),
            'metallic_roughness': ('metallicRoughnessTexture', "Metallic roughness"),
            'normal': ('normalTexture', "Normal"),
            'occlusion': ('occlusionTexture', "Occlusion"),
        }
        
        texture_attr, texture_name = TEXTURE_MAPPING[texture]
        texture_data = getattr(trimesh.visual.material, texture_attr)
        
        if texture_data is None:
            raise ValueError(f"{texture_name} texture not found")
            
        to_tensor = transforms.ToTensor()
        return (to_tensor(texture_data).unsqueeze(0).permute(0, 2, 3, 1).cpu().float(),)
    
class Hy3DSetMeshPBRTextures:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "image": ("IMAGE", ),
                "texture" : (["base_color", "emissive", "metallic_roughness", "normal", "occlusion"], ),
            },
        }

    RETURN_TYPES = ("TRIMESH", )
    RETURN_NAMES = ("trimesh",)
    FUNCTION = "set_textures"
    CATEGORY = "Hunyuan3DWrapper"

    def set_textures(self, trimesh, image, texture):
        from trimesh.visual.material import SimpleMaterial
        if isinstance(trimesh.visual.material, SimpleMaterial):
            log.info("Found SimpleMaterial, Converting to PBRMaterial")
            trimesh.visual.material = trimesh.visual.material.to_pbr()

        
        TEXTURE_MAPPING = {
            'base_color': ('baseColorTexture', "Base color"),
            'emissive': ('emissiveTexture', "Emissive"),
            'metallic_roughness': ('metallicRoughnessTexture', "Metallic roughness"),
            'normal': ('normalTexture', "Normal"),
            'occlusion': ('occlusionTexture', "Occlusion"),
        }
        new_mesh = trimesh.copy()
        texture_attr, texture_name = TEXTURE_MAPPING[texture]
        image_np = (image[0].cpu().numpy() * 255).astype(np.uint8)
        if image_np.shape[2] == 4:  # RGBA
            pil_image = Image.fromarray(image_np, 'RGBA')
        else:  # RGB
            pil_image = Image.fromarray(image_np, 'RGB')
            
        setattr(new_mesh.visual.material, texture_attr, pil_image)
            
        return (new_mesh,)

class Hy3DSetMeshPBRAttributes:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "baseColorFactor": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "emissiveFactor": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "metallicFactor": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "roughnessFactor": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "doubleSided": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("TRIMESH", )
    RETURN_NAMES = ("trimesh",)
    FUNCTION = "set_textures"
    CATEGORY = "Hunyuan3DWrapper"

    def set_textures(self, trimesh, baseColorFactor, emissiveFactor, metallicFactor, roughnessFactor, doubleSided):
        
        new_mesh = trimesh.copy()
        new_mesh.visual.material.baseColorFactor = [baseColorFactor, baseColorFactor, baseColorFactor, 1.0]
        new_mesh.visual.material.emissiveFactor = [emissiveFactor, emissiveFactor, emissiveFactor]
        new_mesh.visual.material.metallicFactor = metallicFactor        
        new_mesh.visual.material.roughnessFactor = roughnessFactor
        new_mesh.visual.material.doubleSided = doubleSided
            
        return (new_mesh,)
    
class Hy3DExportMesh:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "filename_prefix": ("STRING", {"default": "3D/Hy3D"}),
                "file_format": (["glb", "obj", "ply", "stl", "3mf", "dae"],),
            },
            "optional": {
                "save_file": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("glb_path",)
    FUNCTION = "process"
    CATEGORY = "Hunyuan3DWrapper"
    OUTPUT_NODE = True

    def process(self, trimesh, filename_prefix, file_format, save_file=True):
        full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(filename_prefix, folder_paths.get_output_directory())
        output_glb_path = Path(full_output_folder, f'{filename}_{counter:05}_.{file_format}')
        output_glb_path.parent.mkdir(exist_ok=True)
        if save_file:
            trimesh.export(output_glb_path, file_type=file_format)
            relative_path = Path(subfolder) / f'{filename}_{counter:05}_.{file_format}'
        else:
            temp_file = Path(full_output_folder, f'hy3dtemp_.{file_format}')
            trimesh.export(temp_file, file_type=file_format)
            relative_path = Path(subfolder) / f'hy3dtemp_.{file_format}'
        
        return (str(relative_path), )
    
class Hy3DNvdiffrastRenderer:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "trimesh": ("TRIMESH",),
                "render_type": (["textured", "vertex_colors", "normals","depth",],),
                "width": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 16, "tooltip": "Width of the rendered image"}),
                "height": ("INT", {"default": 512, "min": 64, "max": 4096, "step": 16, "tooltip": "Height of the rendered image"}),
                "ssaa": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1, "tooltip": "Super-sampling anti-aliasing"}),
                "num_frames": ("INT", {"default": 30, "min": 1, "max": 1000, "step": 1, "tooltip": "Number of frames to render"}),
                "camera_distance": ("FLOAT", {"default": 2.0, "min": -100.1, "max": 1000.0, "step": 0.01, "tooltip": "Camera distance from the object"}),
                "yaw": ("FLOAT", {"default": 0.0, "min": -90.0, "max": 90.0, "step": 0.01, "tooltip": "Start yaw in radians"}),
                "pitch": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.01, "tooltip": "Start pitch in radians"}),
                "fov": ("FLOAT", {"default": 60.0, "min": 1.0, "max": 179.0, "step": 0.01, "tooltip": "Camera field of view in degrees"}),
                "near": ("FLOAT", {"default": 0.1, "min": 0.001, "max": 1000.0, "step": 0.01, "tooltip": "Camera near clipping plane"}),
                "far": ("FLOAT", {"default": 1000.0, "min": 1.0, "max": 10000.0, "step": 0.01, "tooltip": "Camera far clipping plane"}),
                "pan_x": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.001, "tooltip": "Pan in x direction"}),
                "pan_y": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.001, "tooltip": "Pan in y direction"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK",)
    RETURN_NAMES = ("image", "mask")
    FUNCTION = "render"
    CATEGORY = "Hunyuan3DWrapper"

    def render(self, trimesh, width, height, camera_distance, yaw, pitch, fov, near, far, num_frames, ssaa, render_type, pan_x, pan_y):
        try:
            import nvdiffrast.torch as dr
        except ImportError:
            raise ImportError("nvdiffrast not found. Please install it https://github.com/NVlabs/nvdiffrast")
        try:
            from .utils import rotate_mesh_matrix, yaw_pitch_r_fov_to_extrinsics_intrinsics, intrinsics_to_projection
        except ImportError:
            raise ImportError("utils3d not found. Please install it 'pip install git+https://github.com/EasternJournalist/utils3d.git#egg=utils3d'")
        # Create GL context
        device = mm.get_torch_device()
        glctx = dr.RasterizeCudaContext()
        mesh_copy = trimesh.copy()
        mesh_copy = rotate_mesh_matrix(mesh_copy, 90, 'x')
        mesh_copy = rotate_mesh_matrix(mesh_copy, 180, 'z')

        width, height = width * ssaa, height * ssaa
        aspect_ratio = width / height

        # Get UV coordinates and texture if available
        if hasattr(mesh_copy.visual, 'uv') and hasattr(mesh_copy.visual, 'material'):
            uvs = torch.tensor(mesh_copy.visual.uv, dtype=torch.float32, device=device).contiguous()
            
            # Get texture from material
            if hasattr(mesh_copy.visual.material, 'baseColorTexture'):
                pil_texture = getattr(mesh_copy.visual.material, "baseColorTexture")
            elif hasattr(mesh_copy.visual.material, 'image'):
                pil_texture = getattr(mesh_copy.visual.material, "image")
            pil_texture = pil_texture.transpose(Image.FLIP_TOP_BOTTOM)

            # Convert PIL to tensor [B,C,H,W]
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
            texture = transform(pil_texture).to(device)
            texture = texture.unsqueeze(0).permute(0, 2, 3, 1).contiguous() #need to be contiguous for nvdiffrast
        else:
            log.warning("No texture found")
            # Fallback to vertex colors if no texture
            uvs = None
            texture = None
        
        # Get vertices and faces from trimesh
        vertices = torch.tensor(mesh_copy.vertices, dtype=torch.float32, device=device).unsqueeze(0)
        faces = torch.tensor(mesh_copy.faces, dtype=torch.int32, device=device)
        
        yaws = torch.linspace(yaw, yaw + torch.pi * 2, num_frames) 
        pitches = [pitch] * num_frames
        yaws = yaws.tolist()

        r = camera_distance
        extrinsics, intrinsics = yaw_pitch_r_fov_to_extrinsics_intrinsics(yaws, pitches,  r, fov, aspect_ratio, pan_x, pan_y)
        
        image_list = []
        mask_list = []
        pbar = ProgressBar(num_frames)
        for j, (extr, intr) in tqdm(enumerate(zip(extrinsics, intrinsics)), desc='Rendering', disable=False):
            
            perspective = intrinsics_to_projection(intr, near, far)
            RT = extr.unsqueeze(0)
            full_proj = (perspective @ extr).unsqueeze(0)
            
            # Transform vertices to clip space
            vertices_homo = torch.cat([vertices, torch.ones_like(vertices[..., :1])], dim=-1)
            vertices_camera = torch.bmm(vertices_homo, RT.transpose(-1, -2))
            vertices_clip = torch.bmm(vertices_homo, full_proj.transpose(-1, -2))
            
            # Rasterize with proper shape [batch=1, num_vertices, 4]
            rast_out, _ = dr.rasterize(glctx, vertices_clip, faces, (height, width))
            
            if render_type == "textured":
                if uvs is not None and texture is not None:
                    # Interpolate UV coordinates
                    uv_attr, _= dr.interpolate(uvs.unsqueeze(0), rast_out, faces)
                    
                    # Sample texture using interpolated UVs
                    image = dr.texture(tex=texture, uv=uv_attr)
                    image = dr.antialias(image, rast_out, vertices_clip, faces)
                else:
                    raise Exception("No texture found")
            elif render_type == "vertex_colors":
                # Fallback to vertex color rendering
                vertex_colors = (vertices - vertices.min()) / (vertices.max() - vertices.min())
                image = dr.interpolate(vertex_colors, rast_out, faces)[0]
            elif render_type == "depth":
                depth_values = vertices_camera[..., 2:3].contiguous()
                depth_values = (depth_values - depth_values.min()) / (depth_values.max() - depth_values.min())
                depth_values = 1 - depth_values
                image = dr.interpolate(depth_values, rast_out, faces)[0]
                image = dr.antialias(image, rast_out, vertices_clip, faces)
            elif "normals" in render_type:
                normals_tensor = torch.tensor(mesh_copy.vertex_normals, dtype=torch.float32, device=device).contiguous()
                faces_tensor = torch.tensor(mesh_copy.faces, dtype=torch.int32, device=device).contiguous()
                normal_image_tensors = dr.interpolate(normals_tensor, rast_out, faces_tensor)[0]
                normal_image_tensors = dr.antialias(normal_image_tensors, rast_out, vertices_clip, faces)
                normal_image_tensors = torch.nn.functional.normalize(normal_image_tensors, dim=-1)
                image = (normal_image_tensors + 1) * 0.5

            # Create background color
            background_color = torch.zeros((1, height, width, 3), device=device)
            
            # Get alpha mask from rasterization
            mask = rast_out[..., -1:]
            mask = (mask > 0).float()
            
            # Blend rendered image with background
            image = image * mask + background_color * (1 - mask)

            image_list.append(image)
            mask_list.append(mask)
            
            pbar.update(1)
        import torch.nn.functional as F
        image_out = torch.cat(image_list, dim=0)
        if ssaa > 1:
            image_out = F.interpolate(image_out.permute(0, 3, 1, 2), (width, height), mode='bilinear', align_corners=False, antialias=True)
            image_out = image_out.permute(0, 2, 3, 1)
        mask_out = torch.cat(mask_list, dim=0).squeeze(-1)
     
        
        return (image_out.cpu().float(), mask_out.cpu().float(),)

NODE_CLASS_MAPPINGS = {
    "Hy3DModelLoader": Hy3DModelLoader,
    "Hy3DVAELoader": Hy3DVAELoader,
    "Hy3DGenerateMesh": Hy3DGenerateMesh,
    "Hy3DGenerateMeshMultiView": Hy3DGenerateMeshMultiView,
    "Hy3DExportMesh": Hy3DExportMesh,
    "DownloadAndLoadHy3DDelightModel": DownloadAndLoadHy3DDelightModel,
    "DownloadAndLoadHy3DPaintModel": DownloadAndLoadHy3DPaintModel,
    "Hy3DDelightImage": Hy3DDelightImage,
    "Hy3DRenderMultiView": Hy3DRenderMultiView,
    "Hy3DBakeFromMultiview": Hy3DBakeFromMultiview,
    "Hy3DTorchCompileSettings": Hy3DTorchCompileSettings,
    "Hy3DPostprocessMesh": Hy3DPostprocessMesh,
    "Hy3DLoadMesh": Hy3DLoadMesh,
    "Hy3DUploadMesh": Hy3DUploadMesh,
    "Hy3DCameraConfig": Hy3DCameraConfig,
    "Hy3DMeshUVWrap": Hy3DMeshUVWrap,
    "Hy3DSampleMultiView": Hy3DSampleMultiView,
    "Hy3DMeshVerticeInpaintTexture": Hy3DMeshVerticeInpaintTexture,
    "Hy3DApplyTexture": Hy3DApplyTexture,
    "CV2InpaintTexture": CV2InpaintTexture,
    "Hy3DRenderMultiViewDepth": Hy3DRenderMultiViewDepth,
    "Hy3DGetMeshPBRTextures": Hy3DGetMeshPBRTextures,
    "Hy3DSetMeshPBRTextures": Hy3DSetMeshPBRTextures,
    "Hy3DSetMeshPBRAttributes": Hy3DSetMeshPBRAttributes,
    "Hy3DVAEDecode": Hy3DVAEDecode,
    "Hy3DRenderSingleView": Hy3DRenderSingleView,
    "Hy3DDiffusersSchedulerConfig": Hy3DDiffusersSchedulerConfig,
    "Hy3DIMRemesh": Hy3DIMRemesh,
    "Hy3DBPT": Hy3DBPT,
    "Hy3DMeshInfo": Hy3DMeshInfo,
    "Hy3DFastSimplifyMesh": Hy3DFastSimplifyMesh,
    "Hy3DNvdiffrastRenderer": Hy3DNvdiffrastRenderer,
    "TrimeshToMESH": TrimeshToMESH,
    "MESHToTrimesh": MESHToTrimesh,
    }

NODE_DISPLAY_NAME_MAPPINGS = {
    "Hy3DModelLoader": "Hy3DModelLoader",
    #"Hy3DVAELoader": "Hy3DVAELoader",
    "Hy3DGenerateMesh": "Hy3DGenerateMesh",
    "Hy3DGenerateMeshMultiView": "Hy3DGenerateMeshMultiView",
    "Hy3DExportMesh": "Hy3DExportMesh",
    "DownloadAndLoadHy3DDelightModel": "(Down)Load Hy3D DelightModel",
    "DownloadAndLoadHy3DPaintModel": "(Down)Load Hy3D PaintModel",
    "Hy3DDelightImage": "Hy3DDelightImage",
    "Hy3DRenderMultiView": "Hy3D Render MultiView",
    "Hy3DBakeFromMultiview": "Hy3D Bake From Multiview",
    "Hy3DTorchCompileSettings": "Hy3D Torch Compile Settings",
    "Hy3DPostprocessMesh": "Hy3D Postprocess Mesh",
    "Hy3DLoadMesh": "Hy3D Load Mesh",
    "Hy3DUploadMesh": "Hy3D Upload Mesh",
    "Hy3DCameraConfig": "Hy3D Camera Config",
    "Hy3DMeshUVWrap": "Hy3D Mesh UV Wrap",
    "Hy3DSampleMultiView": "Hy3D Sample MultiView",
    "Hy3DMeshVerticeInpaintTexture": "Hy3D Mesh Vertice Inpaint Texture",
    "Hy3DApplyTexture": "Hy3D Apply Texture",
    "CV2InpaintTexture": "CV2 Inpaint Texture",
    "Hy3DRenderMultiViewDepth": "Hy3D Render MultiView Depth",
    "Hy3DGetMeshPBRTextures": "Hy3D Get Mesh PBR Textures",
    "Hy3DSetMeshPBRTextures": "Hy3D Set Mesh PBR Textures",
    "Hy3DSetMeshPBRAttributes": "Hy3D Set Mesh PBR Attributes",
    "Hy3DVAEDecode": "Hy3D VAE Decode",
    "Hy3DRenderSingleView": "Hy3D Render SingleView",
    "Hy3DDiffusersSchedulerConfig": "Hy3D Diffusers Scheduler Config",
    "Hy3DIMRemesh": "Hy3D Instant-Meshes Remesh",
    "Hy3DBPT": "Hy3D BPT",
    "Hy3DMeshInfo": "Hy3D Mesh Info",
    "Hy3DFastSimplifyMesh": "Hy3D Fast Simplify Mesh",
    "Hy3DNvdiffrastRenderer": "Hy3D Nvdiffrast Renderer",
    "TrimeshToMESH": "Trimesh to MESH",
    "MESHToTrimesh": "MESH to Trimesh",
    }
