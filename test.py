import os
import argparse
import torch as th
import torch.nn.functional as F
import time
from Taper2d import taper2d, antitaper2d
from tqdm import tqdm
import conf_mgt
from utils import yamlread
from guided_diffusion import dist_util

try:
    import ctypes

    libgcc_s = ctypes.CDLL('libgcc_s.so.1')
except:
    pass

from guided_diffusion.script_util import (
    NUM_CLASSES,
    model_and_diffusion_defaults,
    classifier_defaults,
    create_model_and_diffusion,
    create_classifier,
    select_args,
)  # noqa: E402


def toU8(sample):
    if sample is None:
        return sample

    sample = ((sample + 1) * 127.5).clamp(0, 255).to(th.uint8)
    sample = sample.permute(0, 2, 3, 1)

    sample = sample.contiguous()
    sample = sample.detach().cpu().numpy()
    return sample


def main(conf: conf_mgt.Default_Conf):
    print("Start", conf['name'])
    device = dist_util.dev(conf.get('device'))

    model, diffusion = create_model_and_diffusion(
        **select_args(conf, model_and_diffusion_defaults().keys()), conf=conf
    )
    model.load_state_dict(
        dist_util.load_state_dict(os.path.expanduser(
            conf.model_path), map_location="cpu")
    )
    model.to(device)
    if conf.use_fp16:
        model.convert_to_fp16()
    model.eval()

    show_progress = conf.show_progress

    if conf.classifier_scale > 0 and conf.classifier_path:
        print("loading classifier...")
        classifier = create_classifier(
            **select_args(conf, classifier_defaults().keys()))
        classifier.load_state_dict(
            dist_util.load_state_dict(os.path.expanduser(
                conf.classifier_path), map_location="cpu")
        )

        classifier.to(device)
        if conf.classifier_use_fp16:
            classifier.convert_to_fp16()
        classifier.eval()

        def cond_fn(x, t, y=None, gt=None, **kwargs):
            assert y is not None
            with th.enable_grad():
                x_in = x.detach().requires_grad_(True)
                logits = classifier(x_in, t)
                log_probs = F.log_softmax(logits, dim=-1)
                selected = log_probs[range(len(logits)), y.view(-1)]
                return th.autograd.grad(selected.sum(), x_in)[0] * conf.classifier_scale
    else:
        cond_fn = None

    def model_fn(x, t, y=None, gt=None, **kwargs):
        assert y is not None
        return model(x, t, y if conf.class_cond else None, gt=gt)

    print("sampling...")
    all_images = []

    dset = 'eval'

    eval_name = conf.get_default_eval_name()

    dl = conf.get_dataloader(dset=dset, dsName=eval_name)
    epoch = 0
    for batch in iter(dl):
        print('\n', epoch, ":")
        epoch += 1
        for k in batch.keys():
            if isinstance(batch[k], th.Tensor):
                batch[k] = batch[k].to(device)

        model_kwargs = {}

        model_kwargs["gt"] = batch['GT']

        gt_keep_mask = batch.get('gt_keep_mask')
        if gt_keep_mask is not None:
            model_kwargs['gt_keep_mask'] = gt_keep_mask

        batch_size = model_kwargs["gt"].shape[0]

        if conf.cond_y is not None:
            classes = th.ones(batch_size, dtype=th.long, device=device)
            model_kwargs["y"] = classes * conf.cond_y
        else:
            classes = th.randint(
                low=0, high=NUM_CLASSES, size=(batch_size,), device=device
            )
            model_kwargs["y"] = classes

        sample_fn = (
            diffusion.p_sample_loop if not conf.use_ddim else diffusion.ddim_sample_loop
        )

        result = sample_fn(
            model_fn,
            (batch_size, 1, conf.image_size, conf.image_size),
            # (batch_size, 1, 160, 160),
            clip_denoised=conf.clip_denoised,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn,
            device=device,
            progress=show_progress,
            return_all=True,
            conf=conf
        )

        model_kwargs['gt_keep_mask'] = th.zeros_like(model_kwargs['gt_keep_mask'])
        finals = result['sample']
        for i in reversed(range(1)):  # 3-5
            t = th.tensor([i] * model_kwargs["gt"].shape[0], device=device)
            model_kwargs['gt'] = finals
            finals = diffusion.p_sample(model_fn, finals, t,
                                        clip_denoised=True,
                                        denoised_fn=None,
                                        cond_fn=None,
                                        model_kwargs=model_kwargs,
                                        conf=conf,
                                        pred_xstart=None
                                        )['sample']

        def change(arr):
            x = arr.permute(0, 2, 3, 1)
            x = x.contiguous()
            x = x.detach().cpu().numpy()
            return x

        srs = change(result['sample'])
        gts = change(result['gt'])
        lrs = change(result.get('gt') * model_kwargs.get('gt_keep_mask') + (-1) *
                     th.ones_like(result.get('gt')) * (1 - model_kwargs.get('gt_keep_mask')))
        gt_keep_masks = change(model_kwargs.get('gt_keep_mask'))
        finals = change(finals)
        conf.eval_imswrite(
            srs=srs, gts=gts, lrs=lrs, gt_keep_masks=gt_keep_masks,
            img_names=batch['GT_name'], dset=dset, name=eval_name, verify_same=False, finals=finals)

    print("resampling & denoising complete")


if __name__ == "__main__":
    # print(th.cuda.is_available())
    # print(th.cuda.device_count())
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf_path', type=str, default="confs/test_128_denoise.yml")
    args = vars(parser.parse_args())

    conf_arg = conf_mgt.conf_base.Default_Conf()
    conf_arg.update(yamlread(args.get('conf_path')))
    main(conf_arg)
