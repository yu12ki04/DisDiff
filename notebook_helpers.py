from torchvision.datasets.utils import download_url
from ldm.util import instantiate_from_config
import torch
import os
import numpy as np
# todo ?
# from google.colab import files
from IPython.display import Image as ipyimg
import ipywidgets as widgets
from PIL import Image
from numpy import asarray
from einops import rearrange, repeat
import torch, torchvision
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import ismap
import time
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torchvision.utils import make_grid


from ldm.data.dis import Mercaritrain_clip
from ldm.util import log_txt_as_img, exists, default, ismap, isimage, mean_flat, count_params, instantiate_from_config
from ldm.modules.ema import LitEma
from ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution
from ldm.models.autoencoder import VQModelInterface, IdentityFirstStage, AutoencoderKL
from ldm.modules.diffusionmodules.util import make_beta_schedule, extract_into_tensor, noise_like
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.modules.diffusionmodules.util import return_wrap
import copy
import os
import pandas as pd
from omegaconf import OmegaConf


def download_models(mode):

    if mode == "superresolution":
        # this is the small bsr light model
        url_conf = 'https://heibox.uni-heidelberg.de/f/31a76b13ea27482981b4/?dl=1'
        url_ckpt = 'https://heibox.uni-heidelberg.de/f/578df07c8fc04ffbadf3/?dl=1'

        path_conf = 'logs/diffusion/superresolution_bsr/configs/project.yaml'
        path_ckpt = 'logs/diffusion/superresolution_bsr/checkpoints/last.ckpt'

        download_url(url_conf, path_conf)
        download_url(url_ckpt, path_ckpt)

        path_conf = path_conf + '/?dl=1' # fix it
        path_ckpt = path_ckpt + '/?dl=1' # fix it
        return path_conf, path_ckpt
    elif mode == "mercari":
        path_conf = '/content/drive/Shareddrives/mercari/resulsts/logs/mercari/0722/exp_vq_mercari/2024-07-22T09-40-43_s0/configs/2024-07-22T09-40-43-project.yaml'
        path_ckpt = '/content/drive/Shareddrives/mercari/resulsts/logs/mercari/0722/exp_vq_mercari/2024-07-22T09-40-43_s0/checkpoints/last.ckpt'

        return path_conf, path_ckpt
    else:
        raise NotImplementedError


def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    global_step = pl_sd["global_step"]
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return {"model": model}, global_step


def get_model(mode):
    path_conf, path_ckpt = download_models(mode)
    config = OmegaConf.load(path_conf)
    model, step = load_model_from_config(config, path_ckpt)
    return model


def get_custom_cond(mode):
    dest = "data/example_conditioning"

    if mode == "superresolution":
        uploaded_img = files.upload()
        filename = next(iter(uploaded_img))
        name, filetype = filename.split(".") # todo assumes just one dot in name !
        os.rename(f"{filename}", f"{dest}/{mode}/custom_{name}.{filetype}")

    elif mode == "text_conditional":
        w = widgets.Text(value='A cake with cream!', disabled=True)
        display(w)

        with open(f"{dest}/{mode}/custom_{w.value[:20]}.txt", 'w') as f:
            f.write(w.value)

    elif mode == "class_conditional":
        w = widgets.IntSlider(min=0, max=1000)
        display(w)
        with open(f"{dest}/{mode}/custom.txt", 'w') as f:
            f.write(w.value)

    else:
        raise NotImplementedError(f"cond not implemented for mode{mode}")


def get_cond_options(mode):
    path = "data/example_conditioning"
    path = os.path.join(path, mode)
    onlyfiles = [f for f in sorted(os.listdir(path))]
    return path, onlyfiles


def select_cond_path(mode):
    path = "data/example_conditioning"  # todo
    path = os.path.join(path, mode)
    onlyfiles = [f for f in sorted(os.listdir(path))]

    selected = widgets.RadioButtons(
        options=onlyfiles,
        description='Select conditioning:',
        disabled=False
    )
    display(selected)
    selected_path = os.path.join(path, selected.value)
    return selected_path

def get_cond(mode, selected_path, text_condition=None):
    example = dict()
    if mode == "superresolution":
        up_f = 4
        visualize_cond_img(selected_path)
        target_size=(3, 64, 64)

        c = Image.open(selected_path)
        c = torch.unsqueeze(torchvision.transforms.ToTensor()(c), 0)
        c_up = torchvision.transforms.functional.resize(c, size=[up_f * c.shape[2], up_f * c.shape[3]], antialias=True)

        # 画像を指定されたサイズにリシェイプ
        c_up = torchvision.transforms.functional.resize(c_up, size=target_size[1:3], antialias=True)  # (C, H, W)にリサイズ
        c = torchvision.transforms.functional.resize(c, size=target_size[1:3], antialias=True)  # (C, H, W)にリサイズ

        c_up = rearrange(c_up, '1 c h w -> 1 h w c')
        c = rearrange(c, '1 c h w -> 1 h w c')
        c = 2. * c - 1.

        c = c.to(torch.device("cuda"))
        example["LR_image"] = c
        example["image"] = c_up

    elif mode == "mercari":
        up_f = 4
        visualize_cond_img(selected_path)
        target_size=(3, 64, 64)

        c = Image.open(selected_path)
        c = torch.unsqueeze(torchvision.transforms.ToTensor()(c), 0)
        c_up = torchvision.transforms.functional.resize(c, size=[up_f * c.shape[2], up_f * c.shape[3]], antialias=True)

        # 画像を指定されたサイズにリシェイプ
        c_up = torchvision.transforms.functional.resize(c_up, size=target_size[1:3], antialias=True)  # (C, H, W)にリサイズ
        c = torchvision.transforms.functional.resize(c, size=target_size[1:3], antialias=True)  # (C, H, W)にリサイズ

        c_up = rearrange(c_up, '1 c h w -> 1 h w c')
        c = rearrange(c, '1 c h w -> 1 h w c')
        c = 2. * c - 1.

        c = c.to(torch.device("cuda"))
        example["LR_image"] = c
        example["image"] = c_up
        
        # テキスト条件を追加
        example["text"] = text_condition if text_condition is not None else "a photo of clothing"
        
    return example


def visualize_cond_img(path):
    display(ipyimg(filename=path))


def run(model, selected_path, task, custom_steps, text_condition=None, resize_enabled=False, classifier_ckpt=None, global_step=None):
    example = get_cond(task, selected_path, text_condition)

    save_intermediate_vid = False
    n_runs = 1
    masked = False
    guider = None
    ckwargs = None
    mode = 'ddim'
    ddim_use_x0_pred = True
    temperature = 1.
    eta = 1.
    make_progrow = True
    custom_shape = (1, 3, 64, 64)

    height, width = example["image"].shape[1:3]
    split_input = height >= 128 and width >= 128

    if split_input:
        ks = 128
        stride = 64
        vqf = 4  #
        model.split_input_params = {"ks": (ks, ks), "stride": (stride, stride),
                                    "vqf": vqf,
                                    "patch_distributed_vq": True,
                                    "tie_braker": False,
                                    "clip_max_weight": 0.5,
                                    "clip_min_weight": 0.01,
                                    "clip_max_tie_weight": 0.5,
                                    "clip_min_tie_weight": 0.01}
    else:
        if hasattr(model, "split_input_params"):
            delattr(model, "split_input_params")

    invert_mask = False

    x_T = None
    for n in range(n_runs):
        if custom_shape is not None:
            x_T = torch.randn(1, custom_shape[1], custom_shape[2], custom_shape[3]).to(model.device)
            x_T = repeat(x_T, '1 c h w -> b c h w', b=custom_shape[0])
        print(x_T.shape)

        logs = make_convolutional_sample(example, model,
                                         mode=mode, custom_steps=custom_steps,
                                         eta=eta, swap_mode=True , masked=masked,
                                         invert_mask=invert_mask, quantize_x0=False,
                                         custom_schedule=None, decode_interval=10,
                                         resize_enabled=resize_enabled, custom_shape=custom_shape,
                                         temperature=temperature, noise_dropout=0.,
                                         corrector=guider, corrector_kwargs=ckwargs, x_T=x_T, save_intermediate_vid=save_intermediate_vid,
                                         make_progrow=make_progrow,ddim_use_x0_pred=ddim_use_x0_pred
                                         )
    return logs


@torch.no_grad()
def convsample_ddim(model, cond, steps, shape, eta=1.0, callback=None, normals_sequence=None,
                    mask=None, x0=None, quantize_x0=False, img_callback=None,
                    temperature=1., noise_dropout=0., score_corrector=None,
                    corrector_kwargs=None, x_T=None, log_every_t=None
                    ):

    ddim = DDIMSampler(model)
    bs = shape[0]  # dont know where this comes from but wayne
    shape = shape[1:]  # cut batch dim
    print(f"Sampling with eta = {eta}; steps: {steps}")
    samples, intermediates = ddim.sample(steps, batch_size=bs, shape=shape, conditioning=cond, callback=callback,
                                         normals_sequence=normals_sequence, quantize_x0=quantize_x0, eta=eta,
                                         mask=mask, x0=x0, temperature=temperature, verbose=False,
                                         score_corrector=score_corrector,
                                         corrector_kwargs=corrector_kwargs, x_T=x_T)

    return samples, intermediates


@torch.no_grad()
def make_convolutional_sample(batch, model, mode="vanilla", custom_steps=None, eta=1.0, swap_mode=False, masked=False,
                              invert_mask=True, quantize_x0=False, custom_schedule=None, decode_interval=1000,
                              resize_enabled=False, custom_shape=None, temperature=1., noise_dropout=0., corrector=None,
                              corrector_kwargs=None, x_T=None, save_intermediate_vid=False, make_progrow=True,ddim_use_x0_pred=False):
    log = dict()

    z, c, x, xrec, xc = model.get_input(batch, model.first_stage_key,
                                        return_first_stage_outputs=True,
                                        force_c_encode=not (hasattr(model, 'split_input_params')
                                                            and model.cond_stage_key == 'coordinates_bbox'),
                                        return_original_cond=True)
    print("c.shape", c.shape)
    c = c.unsqueeze(0)

    log_every_t = 1 if save_intermediate_vid else None

    if custom_shape is not None:
        z = torch.randn(custom_shape)
        print(f"Generating {custom_shape[0]} samples of shape {custom_shape[1:]}")

    z0 = None

    log["input"] = x
    log["reconstruction"] = xrec

    if ismap(xc):
        log["original_conditioning"] = model.to_rgb(xc)
        if hasattr(model, 'cond_stage_key'):
            log[model.cond_stage_key] = model.to_rgb(xc)

    else:
        log["original_conditioning"] = xc if xc is not None else torch.zeros_like(x)
        if model.cond_stage_model:
            log[model.cond_stage_key] = xc if xc is not None else torch.zeros_like(x)
            if model.cond_stage_key =='class_label':
                log[model.cond_stage_key] = xc[model.cond_stage_key]

    with model.ema_scope("Plotting"):
        t0 = time.time()
        img_cb = None

        sample, intermediates = convsample_ddim(model, c, steps=custom_steps, shape=z.shape,
                                                eta=eta,
                                                quantize_x0=quantize_x0, img_callback=img_cb, mask=None, x0=z0,
                                                temperature=temperature, noise_dropout=noise_dropout,
                                                score_corrector=corrector, corrector_kwargs=corrector_kwargs,
                                                x_T=x_T, log_every_t=log_every_t)
        t1 = time.time()

        if ddim_use_x0_pred:
            sample = intermediates['pred_x0'][-1]

    x_sample = model.decode_first_stage(sample)

    try:
        x_sample_noquant = model.decode_first_stage(sample, force_not_quantize=True)
        log["sample_noquant"] = x_sample_noquant
        log["sample_diff"] = torch.abs(x_sample_noquant - x_sample)
    except:
        pass

    log["sample"] = x_sample
    log["time"] = t1 - t0

    return log


@torch.no_grad()
def visualize_diffusion_process(
    model, 
    batch,
    N=8,
    n_row=8,
    ddim_steps=200,
    ddim_eta=1.0,
    return_keys=None,
    plot_options={
        'denoise_rows': False,
        'progressive_rows': True,
        'diffusion_rows': True,
        'swapped_concepts': False,
        'decoded_xstart': False,
        'swapped_concepts_partial': True
    }
):
    """
    バッチデータからDiffusionプロセスを可視化する関数
    
    Args:
        model: Diffusionモデル
        batch: データローダーからのバッチ
        N: サンプル数
        n_row: 表示する行数
        ddim_steps: DDIMのステップ数
        ddim_eta: DDIMのノイズスケール係数
        return_keys: 返すべきキーのリスト
        plot_options: 可視化オプションの辞書
    """
    log = dict()
    use_ddim = ddim_steps is not None

    # 入力の取得と前処理
    z, c, x, xrec, xc = model.get_input(
        batch, 
        model.first_stage_key,
        return_first_stage_outputs=True,
        force_c_encode=True,
        return_original_cond=True,
        bs=N
    )
    
    N = min(x.shape[0], N)
    n_row = min(x.shape[0], n_row)
    log["inputs"] = x
    log["reconstruction"] = xrec

    # コンディショニングの処理
    if model.model.conditioning_key is not None:
        if hasattr(model.cond_stage_model, "decode"):
            xc = model.cond_stage_model.decode(c)
            log["conditioning"] = xc
        elif model.cond_stage_key in ["caption"]:
            xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["caption"])
            log["conditioning"] = xc
        elif model.cond_stage_key == 'class_label':
            xc = log_txt_as_img((x.shape[2], x.shape[3]), batch["human_label"])
            log['conditioning'] = xc
        elif isimage(xc):
            log["conditioning"] = xc
        if ismap(xc):
            log["original_conditioning"] = model.to_rgb(xc)

    # Diffusion行の生成
    if plot_options['diffusion_rows']:
        diffusion_row = []
        z_start = z[:n_row]
        for t in range(model.num_timesteps):
            if t % model.log_every_t == 0 or t == model.num_timesteps - 1:
                t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                t = t.to(model.device).long()
                noise = torch.randn_like(z_start)
                z_noisy = model.q_sample(x_start=z_start, t=t, noise=noise)
                diffusion_row.append(model.decode_first_stage(z_noisy))

        diffusion_row = torch.stack(diffusion_row)
        diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
        diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
        diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
        log["diffusion_row"] = diffusion_grid

    # サンプリングと追加の可視化
    with model.ema_scope("Plotting"):
        samples, z_denoise_row = model.sample_log(
            cond=c,
            batch_size=N,
            ddim=use_ddim,
            ddim_steps=ddim_steps,
            eta=ddim_eta
        )
        x_samples = model.decode_first_stage(samples)
        log["samples"] = x_samples

    # プログレッシブデノイジング
    if plot_options['progressive_rows']:
        with model.ema_scope("Plotting Progressives"):
            img, progressives = model.progressive_denoising(
                c,
                shape=(model.channels, model.image_size, model.image_size),
                batch_size=N
            )
            prog_row = model._get_denoise_row_from_list(progressives, desc="Progressive Generation")
            log["progressive_row"] = prog_row

    # コンセプトスワッピング
    if plot_options['swapped_concepts']:
        x_samples_list = []
        with model.ema_scope("Plotting Swapping"):
            for cdx in range(model.model.diffusion_model.latent_unit):
                swapped_c = c.clone()
                swapped_c = torch.stack(swapped_c.chunk(model.model.diffusion_model.latent_unit, dim=1), dim=1)
                swapped_c = torch.squeeze(swapped_c)
                swapped_c[:,cdx] = swapped_c[0,cdx][None,:].repeat(c.shape[0],1)
                samples, z_denoise_row = model.sample_log(
                    cond=swapped_c.reshape(c.shape[0],-1),
                    batch_size=N,
                    ddim=use_ddim,
                    ddim_steps=ddim_steps,
                    eta=ddim_eta
                )
                x_samples = model.decode_first_stage(samples)
                x_samples_list.append(x_samples)
            log["samples_swapping"] = torch.cat(x_samples_list, dim=0)

    # 部分的なコンセプトスワッピング
    if plot_options['swapped_concepts_partial']:
        x_samples_list = []
        with model.ema_scope("Plotting Swapping"):
            for cdx in range(model.model.diffusion_model.latent_unit):
                swapped_c = c.clone()
                swapped_c = torch.stack(swapped_c.chunk(model.model.diffusion_model.latent_unit, dim=1), dim=1)
                swapped_c = torch.squeeze(swapped_c)
                swapped_c[:,cdx] = swapped_c[0,cdx][None,:].repeat(c.shape[0],1)
                samples, z_denoise_row = model.sample_log(
                    cond=swapped_c.reshape(c.shape[0],-1),
                    batch_size=N,
                    ddim=use_ddim,
                    ddim_steps=ddim_steps,
                    eta=ddim_eta,
                    sampled_index=np.array([cdx]*N)
                )
                x_samples = model.decode_first_stage(samples)
                x_samples_list.append(x_samples)
            log["samples_swapping_partial"] = torch.cat(x_samples_list, dim=0)

    # デコードされたxstartの可視化
    if plot_options['decoded_xstart']:
        with model.ema_scope("Plotting PredXstart"):
            z_start = z[:n_row]
            diffusion_start = []
            diffusion_full = []
            for cdx in range(model.model.diffusion_model.latent_unit):
                diffusion_row = []
                for t in range(model.num_timesteps):
                    if (t % (model.log_every_t//2) == 0 or t == model.num_timesteps - 1) and t >= model.num_timesteps//2:
                        t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                        t = t.to(model.device).long()
                        noise = torch.randn_like(z_start)
                        z_noisy = model.q_sample(x_start=z_start, t=t, noise=noise)

                        model_out = model.apply_model(
                            z_noisy, 
                            t, 
                            c, 
                            return_ids=False, 
                            sampled_concept=np.array([cdx]*n_row)
                        )
                        eps_pred = model_out.pred + extract_into_tensor(model.ddim_coef, t, x.shape) * model_out.sub_grad
                        x_recon = model.predict_start_from_noise(z_noisy, t=t, noise=eps_pred)
                        diffusion_row.append(model.decode_first_stage(x_recon))

                        if cdx == 0:
                            eps_pred = model_out.pred
                            x_recon = model.predict_start_from_noise(z_noisy, t=t, noise=eps_pred)
                            diffusion_start.append(model.decode_first_stage(x_recon))

                            eps_pred = model_out.pred + extract_into_tensor(model.ddim_coef, t, x.shape) * model_out.out_grad
                            x_recon = model.predict_start_from_noise(z_noisy, t=t, noise=eps_pred)
                            diffusion_full.append(model.decode_first_stage(x_recon))

                diffusion_row = torch.stack(diffusion_row)
                diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
                diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
                diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])
                log[f"predXstart_{cdx}"] = diffusion_grid

                if cdx == 0:
                    # Start predictions
                    diffusion_start = torch.stack(diffusion_start)
                    diffusion_grid = rearrange(diffusion_start, 'n b c h w -> b n c h w')
                    diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
                    diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_start.shape[0])
                    log["predXstart_st"] = diffusion_grid

                    # Full predictions
                    diffusion_full = torch.stack(diffusion_full)
                    diffusion_grid = rearrange(diffusion_full, 'n b c h w -> b n c h w')
                    diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
                    diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_full.shape[0])
                    log["predXstart_fl"] = diffusion_grid

    if return_keys:
        if np.intersect1d(list(log.keys()), return_keys).shape[0] == 0:
            return log
        else:
            return {key: log[key] for key in return_keys}
    return log

def get_mercari_dataloader(
    batch_size=8,
    num_workers=4,
    path=None,
    **kwargs
):
    """
    Mercariデータセットのデータローダーを取得する関数
    
    Args:
        batch_size: バッチサイズ
        num_workers: データ読み込みに使用するワーカー数
        path: データセットのパス（Noneの場合はデフォルトパスを使用）
        **kwargs: その他のパラメータ（image_size以外）
    
    Returns:
        DataLoader: Mercariデータセットのデータローダー
    """
    # image_sizeがkwargsに含まれている場合は削除
    if 'image_size' in kwargs:
        del kwargs['image_size']
    
    # データセットの初期化
    dataset = Mercaritrain_clip(
        path=path,
        **kwargs
    )
    
    # データローダーの作成
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        collate_fn=mercari_collate_fn  # カスタムcollate_fnを追加
    )
    
    return dataloader

def mercari_collate_fn(batch):
    """
    バッチデータを適切な形式に変換するcollate関数
    
    Args:
        batch: データセットから取得したバッチ
        
    Returns:
        dict: 処理済みのバッチデータ
    """
    images = []
    texts = []
    
    for item in batch:
        images.append(item['image'])
        # テキストがtupleの場合は文字列に変換
        if isinstance(item.get('text'), tuple):
            texts.append(' '.join(item['text']))
        else:
            texts.append(item.get('text', ''))
    
    return {
        'image': torch.stack(images),
        'text': texts  # リストとして返す
    }

def get_batch_from_dataloader(dataloader):
    """
    データローダーから1バッチを取得する関数
    
    Args:
        dataloader: データローダー
    
    Returns:
        batch: 1バッチ分のデータ
    """
    return next(iter(dataloader))

# 使用例：
"""
# notebookでの使用方法：

# データローダーの取得
dataloader = get_mercari_dataloader(batch_size=8)

# 1バッチの取得
batch = get_batch_from_dataloader(dataloader)

# 可視化の実行
logs = visualize_diffusion_process(
    model,
    batch,
    N=8,
    n_row=8,
    ddim_steps=200,
    plot_options={
        'progressive_rows': True,
        'diffusion_rows': True,
        'swapped_concepts': True
    }
)
"""