"""Grad-ECLIP-inspired attribution for HiCMAE's pooled video queries.

Reference: https://github.com/Cyang-Zhao/Grad-Eclip
This implements a pooled-query adaptation, not the original CLS-token algorithm.
Native attention is observed with hooks and is never replaced.
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def make_index_record(metadata, trial_index, directory):
    record = dict(metadata)
    record.update(trial_index=trial_index, directory=str(directory))
    return record


def color_overlay(images, masks, opacity=0.45):
    """Jet-style heatmap blend; zero relevance leaves original pixels unchanged."""
    images = np.asarray(images, dtype=np.float32)
    masks = np.clip(np.asarray(masks, dtype=np.float32), 0, 1)
    if not 0 <= opacity <= 1:
        raise ValueError('Overlay opacity must be between 0 and 1')
    # Blue -> cyan -> yellow -> red, without a plotting/OpenCV dependency.
    color = np.stack([np.clip(1.5 - np.abs(4 * masks - center), 0, 1)
                      for center in (3, 2, 1)], axis=-1)
    alpha = (opacity * masks)[..., None]
    blended = images * (1 - alpha) + color * alpha
    return np.rint(np.clip(blended, 0, 1) * 255).astype(np.uint8)


def query_relevance(q, k, v, gradient, chunk_size=64):
    """Sum_i,c grad[i,c] * value[j,c] * minmax_cosine(q_i,k_j), per head.

    All tensors are B,H,N,C. Query chunks bound temporary N x N memory.
    Constant query-key similarity rows receive zero spatial weights.
    """
    q, k = F.normalize(q.float(), dim=-1), F.normalize(k.float(), dim=-1)
    v, gradient = v.float(), gradient.float()
    result = v.new_zeros(v.shape[:3])
    for start in range(0, q.shape[2], chunk_size):
        end = min(start + chunk_size, q.shape[2])
        similarity = q[:, :, start:end] @ k.transpose(-1, -2)
        low = similarity.amin(-1, keepdim=True)
        spatial = (similarity - low) / (similarity.amax(-1, keepdim=True) - low).clamp_min(1e-8)
        channel = gradient[:, :, start:end] @ v.transpose(-1, -2)
        result += (spatial * channel).sum(dim=2)
    return result.sum(dim=1)  # B,N, signed; sum attention heads


class VideoAttentionCapture:
    def __init__(self, encoder):
        self.encoder, self.records, self.handles = encoder, {}, []

    def __enter__(self):
        for index, block in enumerate(self.encoder.blocks):
            attention = block.attn

            def capture_qkv(module, inputs, index=index):
                # Recompute detached QKV using native weights; preserve the native graph.
                with torch.no_grad():
                    x = inputs[0]
                    bias = None
                    if module.q_bias is not None:
                        bias = torch.cat((module.q_bias, torch.zeros_like(module.v_bias), module.v_bias))
                    qkv = F.linear(x, module.qkv.weight, bias)
                    b, n, _ = qkv.shape
                    qkv = qkv.reshape(b, n, 3, module.num_heads, -1).permute(2, 0, 3, 1, 4)
                    self.records[index] = dict(q=qkv[0], k=qkv[1], v=qkv[2])

            def capture_output(module, inputs, index=index):
                # Input to output projection = concatenated native per-head AV outputs.
                self.records[index]['z'] = inputs[0]

            self.handles.append(attention.register_forward_pre_hook(capture_qkv))
            self.handles.append(attention.proj.register_forward_pre_hook(capture_output))
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.records.clear()


def explain_video_loss(model, video, audio, label):
    """Explain one trial; return signed layer/tubelet maps plus prediction metadata."""
    if model.training or video.shape[0] != 1:
        raise ValueError('Explanation requires eval mode and a single trial.')
    with torch.no_grad():
        original_logits = model(video, audio)
    with torch.enable_grad(), VideoAttentionCapture(model.encoder) as capture:
        video = video.detach().requires_grad_(True)
        logits = model(video, audio.detach())
        torch.testing.assert_close(logits.detach(), original_logits, rtol=1e-5, atol=1e-6)
        loss = F.cross_entropy(logits, label)
        records = [capture.records[i] for i in sorted(capture.records)]
        # A pure audio head has no computational path through the video encoder.
        gradients = torch.autograd.grad(loss, [r['z'] for r in records]) if records else ()
        maps = []
        with torch.no_grad():
            for record, gradient in zip(records, gradients):
                b, h, n, c = record['v'].shape
                gradient = gradient.reshape(b, n, h, c).permute(0, 2, 1, 3)
                maps.append(query_relevance(record['q'], record['k'], record['v'], gradient)[0])
            embed = model.encoder.patch_embed
            grid = (video.shape[2] // embed.tubelet_size,
                    video.shape[3] // embed.patch_size[0], video.shape[4] // embed.patch_size[1])
            layers = (torch.stack(maps).reshape(len(maps), *grid) if maps
                      else video.new_zeros((len(model.encoder.blocks), *grid)))
            signed = layers.sum(0)
            if not torch.isfinite(layers).all():
                raise RuntimeError('Nonfinite video relevance')
            metadata = dict(loss=loss.item(), label=label.item(), video_dependency=bool(records),
                            prediction=logits.argmax(1).item(),
                            correct=bool(logits.argmax(1).item() == label.item()),
                            logits=logits[0].detach().cpu().tolist(),
                            forward_max_abs_error=(logits.detach()-original_logits).abs().max().item())
            return signed.cpu(), layers.cpu(), metadata


def frame_masks(positive, frames, height, width, scale):
    """Resize spatially; repeat each tubelet mask across its constituent frames."""
    if frames % positive.shape[0]:
        raise ValueError('Frame count must be divisible by tubelet count')
    masks = F.interpolate(positive[:, None], size=(height, width), mode='bilinear', align_corners=False)[:, 0]
    return (masks / max(float(scale), 1e-12)).clamp(0, 1).repeat_interleave(frames // len(masks), dim=0)


def export_test_explanations(model, loaders, rows, conditions, device, output, limit=0):
    """All conditions share a per-trial positive-relevance scale; save actual inputs."""
    model.eval()
    root = Path(output) / 'explanations'
    root.mkdir(parents=True, exist_ok=True)
    count = len(rows) if limit == 0 else min(limit, len(rows))
    manifest = root / 'index.jsonl'
    # A retry rebuilds the index and overwrites the corresponding trial exports.
    manifest.write_text('')
    for index in range(count):
        trial = rows[index]
        results = {}
        for condition in conditions:
            video, audio, label, sample_index = loaders['test_' + condition].dataset[index]
            signed, layers, metadata = explain_video_loss(
                model, video[None].to(device), audio[None].to(device),
                torch.tensor([label], device=device))
            results[condition] = (video, signed, layers, metadata)
        scale = max(float(item[1].clamp_min(0).max()) for item in results.values())
        for condition, (video, signed, layers, metadata) in results.items():
            destination = root / condition / ('%04d_%s' % (index, Path(trial['video']).stem))
            destination.mkdir(parents=True, exist_ok=True)
            positive = signed.clamp_min(0)
            masks = frame_masks(positive, video.shape[1], video.shape[2], video.shape[3], scale)
            mean = video.new_tensor([.485, .456, .406])[:, None, None, None]
            std = video.new_tensor([.229, .224, .225])[:, None, None, None]
            images = (video * std + mean).clamp(0, 1).permute(1, 2, 3, 0)
            masked = (images * masks[..., None] * 255).round().byte().numpy()
            overlays = color_overlay(images.numpy(), masks.numpy())
            for frame, image in enumerate(masked):
                Image.fromarray(image).save(destination / ('frame_%02d_masked.png' % frame))
                Image.fromarray(overlays[frame]).save(destination / ('frame_%02d_overlay.png' % frame))
            np.savez_compressed(destination / 'heatmaps.npz', signed=signed.numpy(),
                                signed_layers=layers.numpy(), positive=positive.numpy(),
                                normalized=(positive / max(float(positive.max()), 1e-12)).numpy(),
                                shared_scale_masks=masks.numpy())
            metadata.update(trial=trial, condition=condition, shared_positive_scale=scale,
                            attribution='pooled-query Grad-ECLIP adaptation; target=cross entropy',
                            display='positive loss relevance; shared scale across conditions within trial',
                            overlay='jet-style; alpha=0.45*normalized_relevance; zero leaves image unchanged')
            (destination / 'metadata.json').write_text(json.dumps(metadata, indent=2))
            with manifest.open('a') as f:
                f.write(json.dumps(make_index_record(metadata, index, destination)) + '\n')
        print('Saved explanations %d/%d' % (index+1, count), flush=True)


def main():
    """Retry explanations from saved run files without repeating training."""
    import argparse
    import csv
    from types import SimpleNamespace
    from run_ravdess_interaction import make_model, make_loaders, TEST_CONDITIONS

    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument('--output-dir', required=True, help='Existing experiment output folder')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--explain-limit', type=int, default=0)
    options = parser.parse_args()
    if options.explain_limit < 0:
        parser.error('--explain-limit must be nonnegative')
    output = Path(options.output_dir)
    config = json.loads((output / 'config.json').read_text())
    config.update(device=options.device, explain_limit=options.explain_limit)
    args = SimpleNamespace(**config)
    splits = {}
    for name in ('train', 'test'):
        with (output / (name + '.csv')).open(newline='') as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            for key in ('label', 'actor', 'noise_seed'):
                if key in row and row[key] != '':
                    row[key] = int(row[key])
        splits[name] = rows
    checkpoint_path = output / 'final.pth'
    if not checkpoint_path.exists():
        checkpoint_path = output / 'best.pth'  # Backward-compatible old actor-split run.
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    print('Loading explanation checkpoint:', checkpoint_path)
    if any(key.startswith('encoder_fusion.') for key in checkpoint['model']):
        from run_ravdess_fusion import make_model, normalize_heads
        args.architecture = 'fusion'
        saved_heads = {key.split('.')[1] for key in checkpoint['model'] if key.startswith('heads.')}
        if saved_heads:
            args.heads = list(normalize_heads(saved_heads))
            args.logit_terms = list(args.heads)
        else:
            if hasattr(args, 'heads'):
                delattr(args, 'heads')
            args.logit_terms = ['video', 'fusion']
    else:
        args.logit_terms = [term for term, key in (('video', 'wv.weight'), ('audio', 'wa.weight'),
                                                  ('interaction', 'wf.weight')) if key in checkpoint['model']]
    model = make_model(args)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.to(options.device).eval()
    loaders = make_loaders(splits, args)
    export_test_explanations(model, loaders, splits['test'], TEST_CONDITIONS,
                             options.device, output, options.explain_limit)


if __name__ == '__main__':
    main()
