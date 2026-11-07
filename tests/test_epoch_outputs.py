import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


@unittest.skipUnless(all(importlib.util.find_spec(m) for m in ('torch', 'numpy', 'sklearn')),
                     'Requires torch, numpy, scikit-learn')
class EpochOutputsTests(unittest.TestCase):
    def test_training_and_three_test_histories(self):
        import torch
        import run_ravdess_interaction as experiment

        class Dataset(torch.utils.data.Dataset):
            def __init__(self, expanded=False):
                self.rows = [dict(video=str(i), actor=1, label=i) for i in range(6)]
                if expanded:
                    self.rows = experiment.expand_training_rows(self.rows)
            def __len__(self):
                return len(self.rows)
            def __getitem__(self, i):
                label = self.rows[i]['label']
                return torch.eye(6)[label], torch.zeros(6), label, i

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.wv = torch.nn.Linear(6, 6, bias=False)
                self.wf = torch.nn.Linear(6, 6, bias=False)
                self.logit_terms = ('video', 'interaction')
            def forward(self, v, a, return_terms=False):
                terms = (self.wv(v), self.wf(v*a))
                logits = sum(terms)
                return (logits, terms) if return_terms else logits

        dataset = Dataset()
        loaders = {name: torch.utils.data.DataLoader(dataset, batch_size=3)
                   for name in ['train', 'test_both', 'test_video_noise', 'test_audio_neutral']}
        loaders['train'] = torch.utils.data.DataLoader(Dataset(expanded=True), batch_size=3, shuffle=True)
        args = SimpleNamespace(seed=42, device='cpu', checkpoint='unused', lr=.001,
                               epochs=2, skip_epoch_tests=False, skip_explanations=True)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            with patch.object(experiment, 'make_model', return_value=Model()), \
                 patch.object(experiment, 'load_pretrained'), \
                 patch.object(experiment, 'make_loaders', return_value=loaders):
                experiment.train({name: dataset.rows for name in ('train', 'test')}, args, output)
            with (output / 'epoch_metrics.csv').open() as f:
                metrics = list(csv.DictReader(f))
            self.assertEqual(len(metrics), 14)
            for row in metrics:
                if row['split'] == 'train':
                    self.assertEqual(int(row['total']), 18)
                elif row['split'].startswith('train_'):
                    self.assertEqual(int(row['total']), 6)
            self.assertTrue((output / 'final.pth').exists())
            self.assertTrue((output / 'classifier_weights.npz').exists())
            import numpy as np
            with np.load(output / 'classifier_weights.npz') as weights:
                self.assertNotIn('wa', weights.files)
            self.assertFalse((output / 'best.pth').exists())
            self.assertFalse(any(row['split'] == 'val' for row in metrics))
            for row in metrics:
                self.assertAlmostEqual(float(row['accuracy']), int(row['correct']) / int(row['total']))
            self.assertEqual(len(list((output / 'epoch_predictions').glob('*.csv'))), 14)
            summary = json.loads((output / 'test_metrics.json').read_text())
            self.assertEqual(summary['checkpoint_epoch'], 2)
            self.assertEqual(summary['checkpoint_selection'], 'final_epoch')
            self.assertEqual(set(summary['conditions']), set(experiment.TEST_CONDITIONS))
            for condition in experiment.TEST_CONDITIONS:
                with (output / ('test_predictions_' + condition + '.csv')).open() as f:
                    rows = list(csv.DictReader(f))
                self.assertEqual(len(rows), 6)
                for row in rows:
                    self.assertEqual(int(row['correct']), int(row['label'] == row['prediction']))
                    for emotion in experiment.EMOTIONS:
                        self.assertNotIn('audio_logit_' + emotion, row)
                        total = sum(float(row[b + '_logit_' + emotion]) for b in ('video', 'interaction'))
                        self.assertAlmostEqual(total, float(row['logit_' + emotion]), places=5)


if __name__ == '__main__':
    unittest.main()
