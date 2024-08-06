import math
import os
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import torch
from einops import rearrange, repeat
from fire import Fire
from omegaconf import OmegaConf
from PIL import Image
from sgm.util import append_dims, default, instantiate_from_config, freq_mix_3d, freq_mix_4d, get_freq_filter, ScaleScheduler
from torchvision.transforms import ToTensor
from mediapy import write_video
from tqdm import tqdm
from sgm.modules.diffusionmodules.sampling_utils import to_d
from torchvision.utils import save_image
from sgm.modules.attention import TimeSyncMemoryEfficientCrossAttention

import torch.distributed as dist
from torch.multiprocessing import Process, set_start_method, Manager

def parallel_denoising_ddp(rank, world_size, 
                           model_sv3d, model_svd,
                           x_all,
                           sv3d_condition_devices, svd_condition_devices,
                           frame_intervals, view_intervals,
                           num_frames_sv3d, num_frames_svd,
                           num_sigmas, sigmas,
                           ss,
                           manual_cfg, 
                           uc_type, 
                           svd_only_from,
                           svd_cfg_warmup,
                           conv_blend, blend_weight,
                           output_container,
                           ):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
    x_all = x_all.to(rank).contiguous()
    
    cur_sv3d_model = model_sv3d.to(rank)
    cur_svd_model = model_svd.to(rank)
    
    for m in [cur_sv3d_model, cur_svd_model]:
        for p in m.parameters():
            p.requires_grad_(False)
            
    if conv_blend:
        TimeSyncMemoryEfficientCrossAttention.IS_REF_FRAME = False
        TimeSyncMemoryEfficientCrossAttention.BLEND_WEIGHT = blend_weight
    
    cur_cond_sv3d, cur_uc_sv3d = sv3d_condition_devices[rank]
    cur_cond_svd, cur_uc_svd = svd_condition_devices[rank]
    
    cur_cond_sv3d = {k:v.to(rank) for k,v in cur_cond_sv3d.items()}
    cur_uc_sv3d = {k:v.to(rank) for k,v in cur_uc_sv3d.items()}
    cur_cond_svd = {k:v.to(rank) for k,v in cur_cond_svd.items()}
    cur_uc_svd = {k:v.to(rank) for k,v in cur_uc_svd.items()}
    
    frame_idx = frame_intervals[rank]
    frame_idx_end = frame_intervals[rank+1]
    cur_batch_size_sv3d = frame_idx_end - frame_idx
    additional_model_inputs_sv3d = {}
    additional_model_inputs_sv3d["image_only_indicator"] = torch.zeros(
        2 * cur_batch_size_sv3d, num_frames_sv3d
    ).to(rank)
    additional_model_inputs_sv3d["num_video_frames"] = num_frames_sv3d
    
    view_idx = view_intervals[rank]
    view_idx_end = view_intervals[rank+1]
    cur_batch_size_svd = view_idx_end - view_idx
    additional_model_inputs_svd = {}
    additional_model_inputs_svd["image_only_indicator"] = torch.zeros(
        2 * cur_batch_size_svd, num_frames_svd
    ).to(cur_svd_model.device)
    additional_model_inputs_svd["num_video_frames"] = num_frames_svd
    
    cur_slice_sv3d = (0, slice(None), slice(frame_idx, frame_idx_end))
    cur_slice_svd = (1, slice(view_idx, view_idx_end))
    
    for i in tqdm(range(num_sigmas-1)):
        gamma = (
            min(model_svd.sampler.s_churn / (num_sigmas - 1), 2**0.5 - 1)
            if model_svd.sampler.s_tmin <= sigmas[i] <= model_svd.sampler.s_tmax
            else 0.0
        )
        sigma = sigmas[i]
        next_sigma = sigmas[i + 1]
        sigma_hat = sigma * (gamma + 1.0)
        sigma_hat = sigma_hat.to(rank)
        if i < 10:
            num_rollback = 2
        else:
            num_rollback = 1
        for j in range(num_rollback):
            if gamma > 0:
                eps = torch.randn_like(x_all) * model_svd.sampler.s_noise
                x_all = x_all + eps * append_dims(sigma_hat**2 - sigma**2, x_all.ndim) ** 0.5
            
            if not manual_cfg:
                denoised_all = x_all.new_zeros((2,) + x_all.shape)
            else:
                denoised_all = x_all.new_zeros((6,) + x_all.shape)
            
            cur_x_sv3d = rearrange(x_all[:,frame_idx:frame_idx_end].transpose(0,1), "b t ... -> (b t) ...")
            cur_x_svd = rearrange(x_all[view_idx:view_idx_end], "b t ... -> (b t) ...")
            with torch.autocast("cuda"):
                with torch.no_grad():
                    def denoiser_sv3d(input, sigma, c):
                        return cur_sv3d_model.denoiser(
                            cur_sv3d_model.model, input, sigma, c, **additional_model_inputs_sv3d
                        )
                    denoised = denoiser_sv3d(*cur_sv3d_model.sampler.guider.prepare_inputs(
                            cur_x_sv3d, 
                            sigma_hat * sigma_hat.new_ones(cur_batch_size_sv3d * num_frames_sv3d), 
                            cur_cond_sv3d, 
                            cur_uc_sv3d
                    ))
                    x_u, x_c = denoised.chunk(2)
                    x_u = rearrange(x_u, "(b t) ... -> b t ...", t=cur_sv3d_model.sampler.guider.num_frames)
                    x_c = rearrange(x_c, "(b t) ... -> b t ...", t=cur_sv3d_model.sampler.guider.num_frames)
                    scale = repeat(cur_sv3d_model.sampler.guider.scale, "1 t -> b t", b=x_u.shape[0])
                    scale = append_dims(scale, x_u.ndim).to(x_u.device)
                    denoised = x_u + scale * (x_c - x_u)
                    if not manual_cfg:
                        cur_denoised = denoised.transpose(0, 1)
                        denoised_all[cur_slice_sv3d] += cur_denoised
                    else:
                        cur_denoised, cur_denoised_u, cur_denoised_c = denoised.transpose(0, 1), x_u.transpose(0, 1), x_c.transpose(0, 1)
                        denoised_all[cur_slice_sv3d] += cur_denoised
                        denoised_all[(2+cur_slice_sv3d[0],)+cur_slice_sv3d[1:]] += cur_denoised_u
                        denoised_all[(4+cur_slice_sv3d[0],)+cur_slice_sv3d[1:]] += cur_denoised_c
            
                    torch.cuda.empty_cache()
            
                    def denoiser_svd(input, sigma, c):
                        return cur_svd_model.denoiser(
                            cur_svd_model.model, input, sigma, c, **additional_model_inputs_svd
                        )
                    denoised = denoiser_svd(*cur_svd_model.sampler.guider.prepare_inputs(
                            cur_x_svd.to(cur_svd_model.device), 
                            sigma_hat * sigma_hat.new_ones(cur_batch_size_svd * num_frames_svd), 
                            cur_cond_svd, 
                            cur_uc_svd
                        ))
                    x_u, x_c = denoised.chunk(2)
                    x_u = rearrange(x_u, "(b t) ... -> b t ...", t=cur_svd_model.sampler.guider.num_frames)
                    x_c = rearrange(x_c, "(b t) ... -> b t ...", t=cur_svd_model.sampler.guider.num_frames)
                    
                    scale = repeat(cur_svd_model.sampler.guider.scale, "1 t -> b t", b=x_u.shape[0])
                    scale = append_dims(scale, x_u.ndim).to(x_u.device)
                    denoised = x_u + scale * (x_c - x_u)
                    if not manual_cfg:
                        cur_denoised = denoised
                        denoised_all[cur_slice_svd] += cur_denoised
                    else:
                        cur_denoised, cur_denoised_u, cur_denoised_c = denoised, x_u, x_c
                        denoised_all[cur_slice_svd] += cur_denoised
                        denoised_all[(2+cur_slice_svd[0],)+cur_slice_svd[1:]] += cur_denoised_u
                        denoised_all[(4+cur_slice_svd[0],)+cur_slice_svd[1:]] += cur_denoised_c
            
            dist.barrier()
            dist.reduce(denoised_all, dst=0, op=dist.ReduceOp.SUM)
            if rank == 0:
                if manual_cfg:
                    denoised_sv3d_u, denoised_sv3d_c = denoised_all[2], denoised_all[4]
                    denoised_svd_u, denoised_svd_c = denoised_all[3], denoised_all[5]
                    
                    if uc_type == 'sv3d':
                        denoised_svd_u = denoised_sv3d_u
                    if uc_type == 'svd':
                        denoised_sv3d_u = denoised_svd_u
                    if uc_type == 'mean':
                        denoised_svd_u = denoised_sv3d_u = (denoised_sv3d_u + denoised_svd_u) / 2
                        
                    sv3d_cfg_scale = append_dims(cur_sv3d_model.sampler.guider.scale, denoised_sv3d_u.ndim).transpose(0, 1)
                    svd_cfg_scale = append_dims(cur_svd_model.sampler.guider.scale, denoised_svd_u.ndim)
                    if svd_cfg_warmup:
                        svd_cfg_scale_min = torch.ones_like(svd_cfg_scale)
                        svd_cfg_scale = (svd_cfg_scale - svd_cfg_scale_min) * (i / (num_sigmas - 2)) + svd_cfg_scale_min
                    
                    denoised_sv3d = denoised_sv3d_u + sv3d_cfg_scale * (denoised_sv3d_c - denoised_sv3d_u)
                    denoised_svd = denoised_svd_u + svd_cfg_scale * (denoised_svd_c - denoised_svd_u)
                    
                    denoised_all = torch.stack([denoised_sv3d, denoised_svd], dim=0)

                v3d_scale = ss(i)
                
                view_weight = 1 - 0.75 * (1 - x_all.new_tensor([
                    0.91311939, 0.68267051, 0.38873953, 0.13347406, 0.00558459,
                    0.04951557, 0.25      , 0.53736505, 0.8117449 , 0.9777864 ,
                    0.9777864 , 0.8117449 , 0.53736505, 0.25      , 0.04951557,
                    0.00558459, 0.13347406, 0.38873953, 0.68267051, 0.91311939,
                    1.
                ])) # 0.5 * (1 + cos(2 * pi * (x + 1) / 10.5))
                diffusion_weight = torch.stack([1 - view_weight + view_weight * v3d_scale, view_weight * (1 - v3d_scale)], dim=0)
                
                if (i / (num_sigmas - 2)) > svd_only_from:
                    diffusion_weight[0] = 0.
                    diffusion_weight[1] = 1.
                
                denoised_all = (append_dims(diffusion_weight, denoised_all.ndim) * denoised_all).sum(0)
                
                d = to_d(x_all, sigma_hat, denoised_all)
                dt = append_dims(next_sigma - sigma_hat, x_all.ndim)
                x_all += dt * d
                if j < num_rollback - 1:
                    # Langevin dynamic for aproximation of the mofilled distribution of image array at current noise level
                    x_all -= dt * torch.randn(d.shape, device=d.device)
            
            dist.barrier()
            dist.broadcast(x_all, src=0)
        if rank == 0 and i == (num_sigmas - 2):
            output_container.append(x_all.detach().cpu().numpy())
    dist.destroy_process_group()

def sample(
    input_path: str = "assets/test_image.png",  # Can either be image file or folder with image files
    num_steps: Optional[int] = 50,
    fps_id: int = 24,
    motion_bucket_id: int = 127,
    seed: int = 23,
    decoding_t: int = 10,  # Number of frames decoded at a time! This eats most VRAM. Reduce if necessary.
    device: str = "cuda",
    output_folder: Optional[str] = None,
    elevations_deg: Optional[float | List[float]] = 10.0, 
    azimuths_deg: Optional[List[float]] = None, 
    image_frame_ratio: Optional[float] = None, 
    verbose: Optional[bool] = False,
    
    sv3d_scale: float = 0.6,
    sv3d_scale_min: float = 0.0,
    sv3d_scale_max: float = 1.0,
    custom_prefix: str = 'default',
    sv3d_scale_schedule: str = 'constant',
    scale_k: float = 15.0,
    num_iters_stage_1: int = 1,
    num_iters_stage_2: int = 1,
    noise_from_mv: Optional[bool] = False,
    svd_cfg_warmup: Optional[bool] = False,
    uc_type: str = 'default',
    svd_only_from: float = math.inf,
    conv_blend: bool = False,
    blend_weight: float = 0.6,
    world_size: int = 8
):
    """
    Simple script to generate a single sample conditioned on an image `input_path` or multiple images, one for each
    image file in folder `input_path`. If you run out of VRAM, try decreasing `decoding_t`.
    """
    
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    
    num_steps = default(num_steps, 30)
    
    video_specifed = os.path.isdir(input_path) and not os.path.isfile(input_path)
    if not video_specifed:
        output_folder = default(output_folder, f"output/{os.path.basename(input_path).split('.')[0]}")
    else:
        output_folder = default(output_folder, f"output/{os.path.basename(input_path)}")
    output_folder = os.path.join(output_folder, f"{custom_prefix}_seed_{seed:07d}")
    os.makedirs(output_folder, exist_ok=True)
    
    num_frames_svd = 25
    model_config_svd = "configs/svd_xt.yaml"
    cond_aug_svd = 0.02

    num_frames_sv3d = 21
    model_config_sv3d = "configs/sv3d_p.yaml"
    cond_aug_sv3d = 1e-5
    if isinstance(elevations_deg, float) or isinstance(elevations_deg, int):
        elevations_deg = [elevations_deg] * num_frames_sv3d
    assert (
        len(elevations_deg) == num_frames_sv3d
    ), f"Please provide 1 value, or a list of {num_frames_sv3d} values for elevations_deg! Given {len(elevations_deg)}"
    polars_rad = [np.deg2rad(90 - e) for e in elevations_deg]
    if azimuths_deg is None:
        azimuths_deg = np.linspace(0, 360, num_frames_sv3d + 1)[1:] % 360
    assert (
        len(azimuths_deg) == num_frames_sv3d
    ), f"Please provide a list of {num_frames_sv3d} values for azimuths_deg! Given {len(azimuths_deg)}"
    azimuths_rad = [np.deg2rad((a - azimuths_deg[-1]) % 360) for a in azimuths_deg]
    azimuths_rad[:-1].sort()
    
    azimuths_rad[-1] = np.deg2rad(360)
    
    ss = ScaleScheduler(num_steps=num_steps, scale=sv3d_scale, scale_max=sv3d_scale_max, scale_min=sv3d_scale_min, scale_type=sv3d_scale_schedule, scale_k=scale_k)

    model_svd = load_model(
        model_config_svd,
        device,
        num_frames_svd,
        num_steps,
        verbose,
    )
    
    model_sv3d = load_model(
        model_config_sv3d,
        device,
        num_frames_sv3d,
        num_steps,
        verbose,
        conv_blend,
    )

    path = Path(input_path)
    all_img_paths = []
    if path.is_file():
        if any([input_path.endswith(x) for x in ["jpg", "jpeg", "png"]]):
            all_img_paths = [input_path]
        else:
            raise ValueError("Path is not valid image file.")
    elif path.is_dir():
        all_img_paths = sorted(
            [
                f
                for f in path.iterdir()
                if f.is_file() and f.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ]
        )
        all_img_paths = sorted(all_img_paths, key=lambda x: int(x.stem))
        if len(all_img_paths) == 0:
            raise ValueError("Folder does not contain any images.")
    else:
        raise ValueError

    if video_specifed:
        all_img_paths = all_img_paths[:num_frames_svd]
    else:
        all_img_paths = all_img_paths[:1]
        
    input_img_path = all_img_paths[-1]
    
    # Load image
    
    from rembg import remove
    
    image = Image.open(input_img_path)
    if image.mode == "RGBA":
        pass
    else:
        # remove bg
        image.thumbnail([768, 768], Image.Resampling.LANCZOS)
        image = remove(image.convert("RGBA"), alpha_matting=True)

    # resize object in frame
    image_arr = np.array(image)
    in_w, in_h = image_arr.shape[:2]
    ret, mask = cv2.threshold(
        np.array(image.split()[-1]), 0, 255, cv2.THRESH_BINARY
    )
    x, y, w, h = cv2.boundingRect(mask)
    max_size = max(w, h)
    side_len = (
        int(max_size / image_frame_ratio)
        if image_frame_ratio is not None
        else in_w
    )
    padded_image = np.zeros((side_len, side_len, 4), dtype=np.uint8)
    center = side_len // 2
    if video_specifed and image_frame_ratio is None:
        padded_image[:] = image_arr[:]
    else:
        padded_image[
            center - h // 2 : center - h // 2 + h,
            center - w // 2 : center - w // 2 + w,
        ] = image_arr[y : y + h, x : x + w]
    # resize frame to 576x576
    rgba = Image.fromarray(padded_image).resize((576, 576), Image.LANCZOS)
    # white bg
    rgba_arr = np.array(rgba) / 255.0
    rgb = rgba_arr[..., :3] * rgba_arr[..., -1:] + (1 - rgba_arr[..., -1:])
    input_image = Image.fromarray((rgb * 255).astype(np.uint8))

    image = ToTensor()(input_image)
    image = image * 2.0 - 1.0

    image = image.unsqueeze(0).to(device)
    H, W = image.shape[2:]
    assert image.shape[1] == 3
    F = 8
    C = 4
    
    if (H, W) != (576, 576):
        print(
            "WARNING: The conditioning frame you provided is not 576x576. This leads to suboptimal performance as sv3d was only trained on 576x576."
        )
    if motion_bucket_id > 255:
        print(
            "WARNING: High motion bucket! This may lead to suboptimal performance."
        )

    if fps_id < 5:
        print("WARNING: Small fps value! This may lead to suboptimal performance.")

    if fps_id > 30:
        print("WARNING: Large fps value! This may lead to suboptimal performance.")
    
    torch.manual_seed(seed)
    
    # sv3d generation
    shape_sv3d = (num_frames_sv3d, C, H // F, W // F)

    value_dict = {}
    value_dict["cond_frames_without_noise"] = image
    value_dict["motion_bucket_id"] = motion_bucket_id
    value_dict["fps_id"] = fps_id
    value_dict["cond_aug"] = cond_aug_sv3d
    value_dict["cond_frames"] = image + cond_aug_sv3d * torch.randn_like(image)
    value_dict["polars_rad"] = polars_rad
    value_dict["azimuths_rad"] = azimuths_rad
    
    sv3d_filter_shape = [
        1, C, num_frames_sv3d, H // F, W // F
    ]
    
    sv3d_freq_filter = get_freq_filter(
        sv3d_filter_shape, 
        device=device, 
        filter_type='butterworth',
        n=4,
        d_s=0.25,
        d_t=0.25
    )

    with torch.no_grad():
        with torch.autocast(device):
            batch, batch_uc = get_batch(
                get_unique_embedder_keys_from_conditioner(model_sv3d.conditioner),
                value_dict,
                [1, num_frames_sv3d],
                T=num_frames_sv3d,
                device=device,
            )
            c, uc = model_sv3d.conditioner.get_unconditional_conditioning(
                batch,
                batch_uc=batch_uc,
                force_uc_zero_embeddings=[
                    "cond_frames",
                    "cond_frames_without_noise",
                ],
            )

            for k in ["crossattn", "concat"]:
                uc[k] = repeat(uc[k], "b ... -> b t ...", t=num_frames_sv3d)
                uc[k] = rearrange(uc[k], "b t ... -> (b t) ...", t=num_frames_sv3d)
                c[k] = repeat(c[k], "b ... -> b t ...", t=num_frames_sv3d)
                c[k] = rearrange(c[k], "b t ... -> (b t) ...", t=num_frames_sv3d)

            randn = torch.randn(shape_sv3d, device=device)

            for iter in range(num_iters_stage_1):
                if iter == 0:
                    initial_noise = randn.detach().clone()
                else:
                    sigma_0 = model_sv3d.sampler.discretization(num_steps, device=device)[0]
                    z_T = (samples_z + initial_noise * sigma_0) / torch.sqrt(1 + sigma_0 ** 2)
                    z_rand = torch.randn(shape_sv3d, device=device)
                    randn = rearrange(freq_mix_3d(
                        rearrange(z_T.to(dtype=torch.float32), 'n c h w -> 1 c n h w'), 
                        rearrange(z_rand, 'n c h w -> 1 c n h w'),
                        LPF=sv3d_freq_filter
                    ),  '1 c n h w -> n c h w ')
                
                additional_model_inputs_sv3d = {}
                additional_model_inputs_sv3d["image_only_indicator"] = torch.zeros(
                    2, num_frames_sv3d
                ).to(device)
                additional_model_inputs_sv3d["num_video_frames"] = batch["num_video_frames"]

                def denoiser_sv3d(input, sigma, c):
                    return model_sv3d.denoiser(
                        model_sv3d.model, input, sigma, c, **additional_model_inputs_sv3d
                    )

                samples_z = model_sv3d.sampler(denoiser_sv3d, randn, cond=c, uc=uc)
                model_sv3d.en_and_decode_n_samples_a_time = decoding_t
                samples_x = model_sv3d.decode_first_stage(samples_z)
                samples_x[-1:] = value_dict["cond_frames_without_noise"]
                samples_sv3d = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)

        
    # svd generation
    if not video_specifed:
        shape_svd = (num_frames_svd, C, H // F, W // F)

        value_dict = {}
        value_dict["cond_frames_without_noise"] = image
        value_dict["motion_bucket_id"] = motion_bucket_id
        value_dict["fps_id"] = fps_id
        value_dict["cond_aug"] = cond_aug_svd
        value_dict["cond_frames"] = image + cond_aug_svd * torch.randn_like(image)
        
        svd_filter_shape = [
            1, C, num_frames_svd, H // F, W // F
        ]
        
        svd_freq_filter = get_freq_filter(
            svd_filter_shape, 
            device=device, 
            filter_type='butterworth',
            n=4,
            d_s=0.25,
            d_t=0.25
        )

        with torch.no_grad():
            with torch.autocast(device):
                batch, batch_uc = get_batch(
                    get_unique_embedder_keys_from_conditioner(model_svd.conditioner),
                    value_dict,
                    [1, num_frames_svd],
                    T=num_frames_svd,
                    device=device,
                )
                c, uc = model_svd.conditioner.get_unconditional_conditioning(
                    batch,
                    batch_uc=batch_uc,
                    force_uc_zero_embeddings=[
                        "cond_frames",
                        "cond_frames_without_noise",
                    ],
                )

                for k in ["crossattn", "concat"]:
                    uc[k] = repeat(uc[k], "b ... -> b t ...", t=num_frames_svd)
                    uc[k] = rearrange(uc[k], "b t ... -> (b t) ...", t=num_frames_svd)
                    c[k] = repeat(c[k], "b ... -> b t ...", t=num_frames_svd)
                    c[k] = rearrange(c[k], "b t ... -> (b t) ...", t=num_frames_svd)

                randn = torch.randn(shape_svd, device=device)

                for iter in range(num_iters_stage_1):
                    if iter == 0:
                        initial_noise = randn.detach().clone()
                    else:
                        sigma_0 = model_svd.sampler.discretization(num_steps, device=device)[0]
                        z_T = (samples_z + initial_noise * sigma_0) / torch.sqrt(1 + sigma_0 ** 2)
                        z_rand = torch.randn(shape_svd, device=device)
                        randn = rearrange(freq_mix_3d(
                            rearrange(z_T.to(dtype=torch.float32), 'n c h w -> 1 c n h w'), 
                            rearrange(z_rand, 'n c h w -> 1 c n h w'),
                            LPF=svd_freq_filter
                        ),  '1 c n h w -> n c h w ')
                    
                    additional_model_inputs_svd = {}
                    additional_model_inputs_svd["image_only_indicator"] = torch.zeros(
                        2, num_frames_svd
                    ).to(device)
                    additional_model_inputs_svd["num_video_frames"] = batch["num_video_frames"]

                    def denoiser_svd(input, sigma, c):
                        return model_svd.denoiser(
                            model_svd.model, input, sigma, c, **additional_model_inputs_svd
                        )

                    samples_z = model_svd.sampler(denoiser_svd, randn, cond=c, uc=uc)
                    model_svd.en_and_decode_n_samples_a_time = decoding_t
                    samples_x = model_svd.decode_first_stage(samples_z)
                    samples_svd = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)
                    
    else:
        sorted_img_paths = sorted(all_img_paths, key=lambda x: int(x.stem))
        input_images = []
        for input_path in sorted_img_paths:
            # Load image
            image_tmp = Image.open(input_path)
            if image_tmp.mode == "RGBA":
                pass
            else:
                # remove bg
                image_tmp.thumbnail([768, 768], Image.Resampling.LANCZOS)
                image_tmp = remove(image_tmp.convert("RGBA"), alpha_matting=True)

            # resize object in frame
            image_arr = np.array(image_tmp)
            in_w, in_h = image_arr.shape[:2]
            ret, mask = cv2.threshold(
                np.array(image_tmp.split()[-1]), 0, 255, cv2.THRESH_BINARY
            )
            x, y, w, h = cv2.boundingRect(mask)
            max_size = max(w, h)
            side_len = (
                int(max_size / image_frame_ratio)
                if image_frame_ratio is not None
                else in_w
            )
            padded_image = np.zeros((side_len, side_len, 4), dtype=np.uint8)
            center = side_len // 2
            if video_specifed and image_frame_ratio is None:
                padded_image[:] = image_arr[:]
            else:
                padded_image[
                    center - h // 2 : center - h // 2 + h,
                    center - w // 2 : center - w // 2 + w,
                ] = image_arr[y : y + h, x : x + w]
            # resize frame to 576x576
            rgba = Image.fromarray(padded_image).resize((576, 576), Image.LANCZOS)
            # white bg
            rgba_arr = np.array(rgba) / 255.0
            rgb = rgba_arr[..., :3] * rgba_arr[..., -1:] + (1 - rgba_arr[..., -1:])
            input_image = Image.fromarray((rgb * 255).astype(np.uint8))
            image_tmp = ToTensor()(input_image)
            input_images.append(image_tmp)
        samples_svd = torch.stack(input_images, dim=0).to(device)
        
    torch.save(samples_svd.detach().cpu(), os.path.join(output_folder, f"ref_video.pth"))
    torch.save(samples_sv3d.detach().cpu(), os.path.join(output_folder, f"ref_multiview.pth"))
    video_svd = (
            (rearrange(samples_svd, "t c h w -> t h w c") * 255)
            .cpu()
            .numpy()
            .astype(np.uint8)
    )
    write_video(os.path.join(output_folder, f"ref_video.mp4"), video_svd, fps=6)
    video_mv = (
            (rearrange(samples_sv3d, "t c h w -> t h w c") * 255)
            .cpu()
            .numpy()
            .astype(np.uint8)
    )
    write_video(os.path.join(output_folder, f"ref_multiview.mp4"), video_mv, fps=6)
    
    svd_filter_shape = [
        1, C, num_frames_sv3d, num_frames_svd, H // F, W // F
    ]
    svd_freq_filter = get_freq_filter(
        svd_filter_shape, 
        device=device, 
        filter_type='butterworth_4d',
        n=4,
        d_s=0.25,
        d_t=0.25
    )
    
    with torch.no_grad():
        sv3d_conditions = []
        for frame_idx in range(num_frames_svd):
            sv3d_c, sv3d_uc = get_condition(model_sv3d, samples_svd[frame_idx:frame_idx+1] * 2 - 1,
                                        motion_bucket_id, fps_id,
                                        cond_aug_sv3d, num_frames_sv3d,
                                        device,
                                        sv3d=True, polars_rad=polars_rad, azimuths_rad=azimuths_rad)
            sv3d_conditions.append((sv3d_c, sv3d_uc))
            
        svd_conditions = []
        for view_idx in range(num_frames_sv3d):
            svd_c, svd_uc = get_condition(model_svd, samples_sv3d[view_idx:view_idx+1] * 2 - 1,
                                        motion_bucket_id, fps_id,
                                        cond_aug_svd, num_frames_svd,
                                        device)
            svd_conditions.append((svd_c, svd_uc))
        
    shape = (num_frames_sv3d, num_frames_svd, C, H // F, W // F) # 18, 25, 4, 64, 64
    randn = torch.randn(shape, device=device)
    
    if noise_from_mv:
        with torch.no_grad():
            encode_model = model_svd
            encode_model.en_and_decode_n_samples_a_time = decoding_t
            sample_z = encode_model.encode_first_stage(samples_sv3d.to(encode_model.device) * 2 - 1).to(device)
            noised_sample_z = repeat(sample_z, "v ... -> v t ...", t=num_frames_svd)
            sigma_0 = model_svd.sampler.discretization(num_steps, device=device)[0]
            randn = (noised_sample_z + randn * sigma_0) / torch.sqrt(1 + sigma_0 ** 2)

    for iter in range(num_iters_stage_2):
        
        model_sv3d = model_sv3d.to(device)
        model_svd = model_svd.to(device)
        
        if iter == 0:
            initial_noise = randn.detach().clone()
        else:
            sigma_0 = model_svd.sampler.discretization(num_steps, device=device)[0]
            z_T = (x_all.to(device) + initial_noise * sigma_0) / torch.sqrt(1 + sigma_0 ** 2)
            z_rand = torch.randn(shape, device=device)
            randn = rearrange(freq_mix_4d(
                rearrange(z_T.to(dtype=torch.float32), 'v n c h w -> 1 c v n h w'), 
                rearrange(z_rand, 'v n c h w -> 1 c v n h w'),
                LPF=svd_freq_filter
            ),  '1 c v n h w -> v n c h w')
        
        x_all = randn.detach().clone()
        
        for frame_idx in range(num_frames_svd):
            x = x_all[:, frame_idx].clone()
            cond, uc = sv3d_conditions[frame_idx]
            x, s_in_v3d, sigmas, num_sigmas, cond, uc = model_sv3d.sampler.prepare_sampling_loop(
                x, cond, uc, num_steps
            )
            sv3d_conditions[frame_idx] = (cond, uc)
            
        for view_idx in range(num_frames_sv3d):
            x = x_all[view_idx]
            cond, uc = svd_conditions[view_idx]
            x, s_in_svd, sigmas, num_sigmas, cond, uc = model_svd.sampler.prepare_sampling_loop(
                x, cond, uc, num_steps
            )
            svd_conditions[view_idx] = (cond, uc)
            
        assert (s_in_svd == 1).all() and (s_in_v3d == 1).all()
            
        frame_intervals = np.cumsum([0,3,3,3,3,3,3,3,4])
        view_intervals = np.cumsum([0,2,3,3,3,3,3,3,1])
        sv3d_condition_devices = [] 
        svd_condition_devices = []
        for gpu_idx in range(world_size):
            frame_idx = frame_intervals[gpu_idx]
            frame_idx_end = frame_intervals[gpu_idx+1]
            cond, uc = dict(crossattn=[], vector=[], concat=[]), dict(crossattn=[], vector=[], concat=[])
            for f in range(frame_idx, frame_idx_end):
                tmp_cond, tmp_uc = sv3d_conditions[f]
                for k in ['crossattn', 'vector', 'concat']:
                    cond[k].append(tmp_cond[k])
                    uc[k].append(tmp_uc[k])
            for k in ['crossattn', 'vector', 'concat']:
                cond[k] = torch.cat(cond[k], dim=0).cpu()
                uc[k] = torch.cat(uc[k], dim=0).cpu()
            sv3d_condition_devices.append((cond, uc))
            
            view_idx = view_intervals[gpu_idx]
            view_idx_end = view_intervals[gpu_idx+1]
            cond, uc = dict(crossattn=[], vector=[], concat=[]), dict(crossattn=[], vector=[], concat=[])
            for f in range(view_idx, view_idx_end):
                tmp_cond, tmp_uc = svd_conditions[f]
                for k in ['crossattn', 'vector', 'concat']:
                    cond[k].append(tmp_cond[k])
                    uc[k].append(tmp_uc[k])
            for k in ['crossattn', 'vector', 'concat']:
                cond[k] = torch.cat(cond[k], dim=0).cpu()
                uc[k] = torch.cat(uc[k], dim=0).cpu()
            svd_condition_devices.append((cond, uc))
        
        model_sv3d = model_sv3d.cpu()
        model_svd = model_svd.cpu()
        svd_ae_model = model_svd.first_stage_model
        model_svd.first_stage_model = None
        x_all = x_all.cpu()
        sigmas = sigmas.cpu().contiguous()
        
        torch.cuda.empty_cache()
        
        manager = Manager()
        output_container = manager.list()
        
        processes = []
        for rank in range(world_size):
            p = Process(target=parallel_denoising_ddp, args=(rank, world_size, 
                           model_sv3d, model_svd,
                           x_all,
                           sv3d_condition_devices, svd_condition_devices,
                           frame_intervals, view_intervals,
                           num_frames_sv3d, num_frames_svd,
                           num_sigmas, sigmas,
                           ss,
                           svd_cfg_warmup or (uc_type != 'default'), 
                           uc_type,
                           svd_only_from,
                           svd_cfg_warmup,
                           conv_blend, blend_weight,
                           output_container))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
            
        model_svd.first_stage_model = svd_ae_model
            
    assert len(output_container) > 0
    x_all = torch.from_numpy(output_container[0]).cuda()
        
    torch.cuda.empty_cache()
    decode_model = model_svd.cuda()
    samples_all = torch.zeros((num_frames_sv3d, num_frames_svd, 3, H, W), device=decode_model.device)
    for view_idx in range(num_frames_sv3d):
        samples_z = x_all[view_idx].to(decode_model.device)
        decode_model.en_and_decode_n_samples_a_time = decoding_t
        samples_x = decode_model.decode_first_stage(samples_z)
        samples = torch.clamp((samples_x + 1.0) / 2.0, min=0.0, max=1.0)
        samples_all[view_idx] = samples
    
    monocular_video = samples_all[
        [i if i < num_frames_sv3d else num_frames_sv3d - 1 for i in range(num_frames_svd)],
        [i for i in range(num_frames_svd)]]
    monocular_video = (
                (rearrange(monocular_video, "t c h w -> t h w c") * 255)
                .cpu()
                .numpy()
                .astype(np.uint8)
            )
    write_video(os.path.join(output_folder, f"preview.mp4"), monocular_video, fps=6)
    
    for view_idx in range(num_frames_sv3d):
        view_video = samples_all[view_idx]
        view_folder = os.path.join(output_folder, f"view_{view_idx:02d}")
        os.makedirs(view_folder, exist_ok=True)
        video_path = os.path.join(view_folder, f"frame_all.mp4")
        frames = (
                (rearrange(view_video, "t c h w -> t h w c") * 255)
                .cpu()
                .numpy()
                .astype(np.uint8)
            )
        write_video(video_path, frames, fps=6)
        for frame_idx, frame in enumerate(frames):
            image_fn = os.path.join(view_folder, f"frame_{frame_idx:02d}.png")
            Image.fromarray(frame).save(image_fn)
                
def get_condition(model, image, motion_bucket_id, fps_id, cond_aug, num_frames, device, sv3d=False, polars_rad=None, azimuths_rad=None):
    value_dict = {}
    value_dict["cond_frames_without_noise"] = image
    value_dict["motion_bucket_id"] = motion_bucket_id
    value_dict["fps_id"] = fps_id
    value_dict["cond_aug"] = cond_aug
    value_dict["cond_frames"] = image + cond_aug * torch.randn_like(image)
    if sv3d:
        value_dict["polars_rad"] = polars_rad
        value_dict["azimuths_rad"] = azimuths_rad
    batch, batch_uc = get_batch(
        get_unique_embedder_keys_from_conditioner(model.conditioner),
        value_dict,
        [1, num_frames],
        T=num_frames,
        device=device,
    )
    c, uc = model.conditioner.get_unconditional_conditioning(
        batch,
        batch_uc=batch_uc,
        force_uc_zero_embeddings=[
            "cond_frames",
            "cond_frames_without_noise",
        ],
    )

    for k in ["crossattn", "concat"]:
        uc[k] = repeat(uc[k], "b ... -> b t ...", t=num_frames)
        uc[k] = rearrange(uc[k], "b t ... -> (b t) ...", t=num_frames)
        c[k] = repeat(c[k], "b ... -> b t ...", t=num_frames)
        c[k] = rearrange(c[k], "b t ... -> (b t) ...", t=num_frames)
        
    return c, uc


def get_unique_embedder_keys_from_conditioner(conditioner):
    return list(set([x.input_key for x in conditioner.embedders]))


def get_batch(keys, value_dict, N, T, device):
    batch = {}
    batch_uc = {}

    for key in keys:
        if key == "fps_id":
            batch[key] = (
                torch.tensor([value_dict["fps_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "motion_bucket_id":
            batch[key] = (
                torch.tensor([value_dict["motion_bucket_id"]])
                .to(device)
                .repeat(int(math.prod(N)))
            )
        elif key == "cond_aug":
            batch[key] = repeat(
                torch.tensor([value_dict["cond_aug"]]).to(device),
                "1 -> b",
                b=math.prod(N),
            )
        elif key == "cond_frames" or key == "cond_frames_without_noise":
            batch[key] = repeat(value_dict[key], "1 ... -> b ...", b=N[0])
        elif key == "polars_rad" or key == "azimuths_rad":
            batch[key] = torch.tensor(value_dict[key]).to(device).repeat(N[0])
        else:
            batch[key] = value_dict[key]

    if T is not None:
        batch["num_video_frames"] = T

    for key in batch.keys():
        if key not in batch_uc and isinstance(batch[key], torch.Tensor):
            batch_uc[key] = torch.clone(batch[key])
    return batch, batch_uc

def load_model(
    config: str,
    device: str,
    num_frames: int,
    num_steps: int,
    verbose: bool = False,
    conv_blend: bool = False,
):
    config = OmegaConf.load(config)
    if device == "cuda":
        config.model.params.conditioner_config.params.emb_models[
            0
        ].params.open_clip_embedding_config.params.init_device = device

    config.model.params.sampler_config.params.verbose = verbose
    config.model.params.sampler_config.params.num_steps = num_steps
    config.model.params.sampler_config.params.guider_config.params.num_frames = (
        num_frames
    )
    if conv_blend:
        config.model.params.network_config.params.spatial_transformer_attn_type = "softmax-xformers-timesync"
    if device == "cuda":
        with torch.device(device):
            model = instantiate_from_config(config.model).to(device).eval()
    else:
        model = instantiate_from_config(config.model).to(device).eval()
        
    return model

if __name__ == "__main__":
    Fire(sample)