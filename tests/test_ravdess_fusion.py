import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class HeadSelectionTests(unittest.TestCase):
    def test_selection_aliases_order_and_errors(self):
        from run_ravdess_fusion import normalize_heads, HEAD_NAMES
        self.assertEqual(normalize_heads(['fusion_audio', 'visual', 'audio']),
                         ('video', 'audio', 'fusion_audio'))
        self.assertEqual(normalize_heads(HEAD_NAMES), HEAD_NAMES)
        for names in ([], ['visual', 'video'], ['audio', 'audio'], ['fusion']):
            with self.assertRaises(ValueError):
                normalize_heads(names)


@unittest.skipUnless(all(importlib.util.find_spec(m) for m in ('torch', 'timm', 'einops')),
                     'Requires upstream PyTorch model dependencies')
class FusionModelTests(unittest.TestCase):
    def test_formula_audio_and_fusion_gradients_and_checkpoint_validation(self):
        import torch
        from torch import nn
        from run_ravdess_fusion import make_model, load_pretrained
        args = SimpleNamespace(embed_dim=384, depth=1, fusion_depth=1, drop_path=0.,
                               logit_terms=['video', 'fusion'])
        model = make_model(args).eval()
        self.assertEqual(model.logit_terms, ('video', 'fusion'))
        self.assertFalse(hasattr(model, 'wa'))
        self.assertEqual(model.wf.in_features, 768)

        class Encoder(nn.Module):
            def __init__(self, tokens):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(1, tokens, 1, 384))
            def forward(self, x):
                return self.weight.expand(x.shape[0], -1, -1, -1)

        model.encoder, model.encoder_audio = Encoder(3), Encoder(4)
        v, a = torch.ones(2, 1), torch.ones(2, 1)
        logits, terms = model(v, a, return_terms=True)
        vl, al = model.encoder(v), model.encoder_audio(a)
        fv, fa = model.encoder_fusion(vl[:, :, -1], al[:, :, -1])
        expected = model.wv(vl.mean(1).squeeze(1)) + model.wf(torch.cat((fv.mean(1), fa.mean(1)), -1))
        torch.testing.assert_close(logits, expected)
        torch.testing.assert_close(logits, terms[0] + terms[1])
        torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
        self.assertGreater(model.encoder_audio.weight.grad.abs().sum().item(), 0)
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                            for p in model.encoder_fusion.parameters()))
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / 'pretrained.pth'
            state = model.state_dict()
            torch.save({'model': {k: val for k, val in state.items()
                                 if not k.startswith('encoder_fusion.')}}, checkpoint)
            with self.assertRaises(ValueError):
                load_pretrained(model, checkpoint)
            head = model.wv.weight.detach().clone()
            state['wv.weight'] = torch.ones_like(state['wv.weight']) * 99
            torch.save({'model': state}, checkpoint)
            load_pretrained(model, checkpoint)
            torch.testing.assert_close(model.wv.weight, head)

    def test_every_head_combination_and_audio_only_zero_video_map(self):
        import itertools
        import torch
        from torch import nn
        from run_ravdess_fusion import make_model, HEAD_NAMES
        from video_loss_explanations import explain_video_loss

        class Encoder(nn.Module):
            def __init__(self, tokens):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(1, tokens, 1, 384))
                self.blocks = nn.ModuleList([])
                self.patch_embed = SimpleNamespace(tubelet_size=2, patch_size=(16, 16))
            def forward(self, x):
                return self.weight.expand(x.shape[0], -1, -1, -1)

        class Fusion(nn.Module):
            def __init__(self):
                super().__init__()
                self.projection = nn.Linear(384, 384)
            def forward(self, video, audio):
                return (video + self.projection(audio.mean(1))[:, None],
                        audio + self.projection(video.mean(1))[:, None])

        def backbone(**kwargs):
            return SimpleNamespace(encoder=Encoder(3), encoder_audio=Encoder(4),
                                   encoder_fusion=Fusion(),
                                   video_layer_weights=nn.Parameter(torch.ones(1)),
                                   audio_layer_weights=nn.Parameter(torch.ones(1)))

        for n in range(1, 5):
            for selected in itertools.combinations(HEAD_NAMES, n):
                args = SimpleNamespace(embed_dim=384, depth=1, fusion_depth=1, drop_path=0., heads=selected)
                with patch('modeling_finetune_av.AudioVisionTransformer', side_effect=backbone):
                    model = make_model(args).eval()
                self.assertEqual(tuple(model.heads), selected)
                v, a = torch.ones(2, 1), torch.ones(2, 1)
                logits, terms = model(v, a, return_terms=True)
                torch.testing.assert_close(logits, sum(terms))
                self.assertEqual(len(terms), n)
                torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
                needs_fusion = any(name.startswith('fusion_') for name in selected)
                self.assertEqual(model.encoder.weight.grad is not None, 'video' in selected or needs_fusion)
                self.assertEqual(model.encoder_audio.weight.grad is not None, 'audio' in selected or needs_fusion)
                for head in model.heads.values():
                    self.assertIsNotNone(head.weight.grad)
                if selected == ('audio',):
                    signed, layers, metadata = explain_video_loss(
                        model, torch.ones(1, 3, 2, 16, 16), torch.ones(1, 1), torch.tensor([0]))
                    self.assertEqual(signed.abs().sum().item(), 0)
                    self.assertFalse(metadata['video_dependency'])


if __name__ == '__main__':
    unittest.main()
