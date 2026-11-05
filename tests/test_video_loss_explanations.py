import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


@unittest.skipUnless(all(importlib.util.find_spec(m) for m in ('torch', 'numpy', 'PIL')),
                     'Requires torch, numpy, Pillow')
class AttributionTests(unittest.TestCase):
    def test_chunked_query_aggregation(self):
        import torch
        import torch.nn.functional as F
        from video_loss_explanations import query_relevance
        torch.manual_seed(42)
        q, k, v, gradient = [torch.randn(1, 2, 5, 3) for _ in range(4)]
        similarity = F.normalize(q, dim=-1) @ F.normalize(k, dim=-1).transpose(-1, -2)
        spatial = (similarity - similarity.amin(-1, keepdim=True)) / (
            similarity.amax(-1, keepdim=True) - similarity.amin(-1, keepdim=True)).clamp_min(1e-8)
        expected = torch.einsum('bhic,bhjc,bhij->bj', gradient, v, spatial)
        torch.testing.assert_close(query_relevance(q, k, v, gradient, chunk_size=2), expected)
        zeros = query_relevance(torch.zeros_like(q), torch.zeros_like(k), v, gradient)
        self.assertTrue(torch.isfinite(zeros).all())
        self.assertEqual(zeros.abs().sum().item(), 0)

    def test_spatial_masks_repeat_without_temporal_interpolation(self):
        import torch
        from video_loss_explanations import frame_masks
        positive = torch.tensor([[[0., 1.], [1., 0.]], [[2., 0.], [0., 2.]]])
        masks = frame_masks(positive, 4, 8, 8, 2)
        self.assertEqual(tuple(masks.shape), (4, 8, 8))
        torch.testing.assert_close(masks[0], masks[1])
        torch.testing.assert_close(masks[2], masks[3])
        self.assertGreater((masks[0]-masks[2]).abs().sum().item(), 0)
        self.assertEqual(frame_masks(torch.zeros_like(positive), 4, 8, 8, 0).sum().item(), 0)

    @unittest.skipUnless(all(importlib.util.find_spec(m) for m in ('timm', 'einops')),
                         'Requires upstream attention dependencies')
    def test_native_forward_capture_and_export(self):
        import numpy as np
        import torch
        from torch import nn
        from modeling_finetune_av import Attention
        from video_loss_explanations import explain_video_loss, export_test_explanations

        class Block(nn.Module):
            def __init__(self):
                super().__init__()
                self.attn = Attention(16, num_heads=2, qkv_bias=True)
                self.norm = nn.LayerNorm(16)
            def forward(self, x):
                return x + self.attn(self.norm(x))

        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.patch_embed = nn.Conv3d(3, 16, (2, 2, 2), (2, 2, 2))
                self.patch_embed.tubelet_size = 2
                self.patch_embed.patch_size = (2, 2)
                self.blocks = nn.ModuleList([Block(), Block()])
            def forward(self, x):
                x = self.patch_embed(x).flatten(2).transpose(1, 2)
                layers = []
                for block in self.blocks:
                    x = block(x)
                    layers.append(x)
                return torch.stack(layers, dim=2).mean((1, 2))

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = Encoder()
                self.head = nn.Linear(16, 6)
            def forward(self, v, a):
                ov = self.encoder(v)
                return self.head(ov + a + ov * a)

        torch.manual_seed(5)
        model = Model().eval()
        video, audio = torch.randn(1, 3, 4, 4, 4), torch.randn(1, 16)
        before = model(video, audio).detach()
        signed, layers, meta = explain_video_loss(model, video, audio, torch.tensor([2]))
        torch.testing.assert_close(before, model(video, audio).detach())
        self.assertEqual(tuple(signed.shape), (2, 2, 2))
        self.assertEqual(tuple(layers.shape), (2, 2, 2, 2))
        torch.testing.assert_close(signed, layers.sum(0))
        self.assertEqual(meta['forward_max_abs_error'], 0.)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        for block in model.encoder.blocks:
            self.assertFalse(block.attn._forward_pre_hooks)
            self.assertFalse(block.attn.proj._forward_pre_hooks)

        class Dataset:
            def __init__(self, v):
                self.v = v
            def __getitem__(self, index):
                return self.v[0], audio[0], 2, index

        conditions = ('both', 'video_noise', 'audio_neutral')
        loaders = {'test_' + c: SimpleNamespace(dataset=Dataset(video)) for c in conditions}
        with tempfile.TemporaryDirectory() as directory:
            rows = [dict(video='02-01-05-01-01-01-01.mp4', actor=1, label=2)]
            export_test_explanations(model, loaders, rows, conditions, 'cpu', directory)
            for condition in conditions:
                folder = next((Path(directory) / 'explanations' / condition).iterdir())
                self.assertEqual(len(list(folder.glob('*_masked.png'))), 4)
                maps = np.load(folder / 'heatmaps.npz')
                self.assertTrue(np.isfinite(maps['shared_scale_masks']).all())
                self.assertTrue((folder / 'metadata.json').exists())


if __name__ == '__main__':
    unittest.main()
