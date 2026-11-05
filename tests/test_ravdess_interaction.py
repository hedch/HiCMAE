import importlib.util
import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from run_ravdess_interaction import balanced_split, discover, make_model


class DataTests(unittest.TestCase):
    def setUp(self):
        self.rows = [dict(actor=a, label=c, video=f'{a}-{c}-{i}', audio='x')
                     for a in range(1, 25) for c in range(8)
                     for i in range(4 if c == 0 else 8)]

    def test_balanced_disjoint_reproducible(self):
        splits, report = balanced_split(self.rows, 42)
        self.assertEqual((splits, report), balanced_split(self.rows, 42))
        self.assertEqual([len(splits[n]) for n in ('train', 'val', 'test')], [512, 128, 128])
        groups = [{r['actor'] for r in splits[n]} for n in ('train', 'val', 'test')]
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        for rows in splits.values():
            self.assertEqual(len({r['video'] for r in rows}), len(rows))
            self.assertEqual(len({sum(r['label'] == c for r in rows) for c in range(8)}), 1)

    def test_invalid_actor_split(self):
        with self.assertRaises(ValueError):
            balanced_split(self.rows, 42, [1, 2], [2, 3])

    def test_pairing_and_missing_audio(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for modality, ext in [('01', 'mp4'), ('02', 'mp4'), ('03', 'wav')]:
                (root / f'{modality}-01-05-01-01-01-01.{ext}').touch()
            rows = discover(root, root)
            self.assertEqual(len(rows), 1)
            self.assertTrue(Path(rows[0]['video']).name.startswith('02-'))
            self.assertEqual(rows[0]['label'], 4)
            (root / '03-01-05-01-01-01-01.wav').unlink()
            with self.assertRaises(ValueError):
                discover(root, root)


@unittest.skipUnless(all(importlib.util.find_spec(m) for m in ('torch', 'timm', 'einops')),
                     'Requires upstream model dependencies')
class ModelTests(unittest.TestCase):
    def test_formula_and_gradients(self):
        import torch
        args = SimpleNamespace(embed_dim=384, depth=1, drop_path=0.)
        model = make_model(args)
        # Tiny deterministic encoder doubles isolate pooling and interaction head.
        class Encoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.ones(()))
            def forward(self, x):
                return x * self.scale
        model.encoder, model.encoder_audio = Encoder(), Encoder()
        self.assertFalse(any('fusion' in name for name, _ in model.named_modules()))
        v, a = torch.randn(2, 3, 1, 384), torch.randn(2, 4, 1, 384)
        logits = model(v, a)
        ov, oa = v.mean(1).squeeze(1), a.mean(1).squeeze(1)
        expected = model.wv(ov) + model.wa(oa) + model.wf(ov * oa)
        torch.testing.assert_close(logits, expected)
        torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1])).backward()
        for name, param in model.named_parameters():
            self.assertIsNotNone(param.grad, name)
            self.assertTrue(torch.isfinite(param.grad).all(), name)


if __name__ == '__main__':
    unittest.main()
