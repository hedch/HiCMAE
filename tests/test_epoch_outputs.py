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
            def __init__(self):
                self.rows = [dict(video=str(i), actor=1, label=i) for i in range(6)]
            def __len__(self):
                return 6
            def __getitem__(self, i):
                return torch.eye(6)[i], torch.zeros(6), i, i

        class Model(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.head = torch.nn.Linear(6, 6)
            def forward(self, v, a):
                return self.head(v + a)

        dataset = Dataset()
        loaders = {name: torch.utils.data.DataLoader(dataset, batch_size=3)
                   for name in ['train', 'val', 'test_both', 'test_video_noise', 'test_audio_neutral']}
        args = SimpleNamespace(seed=42, device='cpu', checkpoint='unused', lr=.001,
                               epochs=2, skip_epoch_tests=False, skip_explanations=True)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            with patch.object(experiment, 'make_model', return_value=Model()), \
                 patch.object(experiment, 'load_pretrained'), \
                 patch.object(experiment, 'make_loaders', return_value=loaders):
                experiment.train({name: dataset.rows for name in ('train', 'val', 'test')}, args, output)
            with (output / 'epoch_metrics.csv').open() as f:
                metrics = list(csv.DictReader(f))
            self.assertEqual(len(metrics), 10)
            for row in metrics:
                self.assertAlmostEqual(float(row['accuracy']), int(row['correct']) / int(row['total']))
            self.assertEqual(len(list((output / 'epoch_predictions').glob('*.csv'))), 10)
            summary = json.loads((output / 'test_metrics.json').read_text())
            self.assertIn(summary['best_epoch'], (1, 2))
            self.assertEqual(set(summary['conditions']), set(experiment.TEST_CONDITIONS))
            for condition in experiment.TEST_CONDITIONS:
                with (output / ('test_predictions_' + condition + '.csv')).open() as f:
                    rows = list(csv.DictReader(f))
                self.assertEqual(len(rows), 6)
                for row in rows:
                    self.assertEqual(int(row['correct']), int(row['label'] == row['prediction']))


if __name__ == '__main__':
    unittest.main()
