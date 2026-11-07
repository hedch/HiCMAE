import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace


@unittest.skipUnless(all(importlib.util.find_spec(m) for m in ('torch', 'timm', 'einops')),
                     'Requires upstream PyTorch model dependencies')
class FusionModelTests(unittest.TestCase):
    def test_formula_audio_and_fusion_gradients_and_checkpoint_validation(self):
        import torch
        from torch import nn
        from run_ravdess_fusion import make_model, load_pretrained
        args = SimpleNamespace(embed_dim=384, depth=1, fusion_depth=1, drop_path=0.)
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


if __name__ == '__main__':
    unittest.main()
