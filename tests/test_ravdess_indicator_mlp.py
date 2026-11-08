import ast
import csv
import importlib.util
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import run_ravdess_indicator_mlp as experiment


class IndicatorDataTests(unittest.TestCase):
    def test_training_and_testing_indicators(self):
        self.assertEqual(experiment.indicators_for_conditions(experiment.TEST_CONDITIONS, True),
                         [[1, 1], [1, 0], [0, 1]])
        self.assertEqual(experiment.indicators_for_conditions(experiment.TEST_CONDITIONS, False),
                         [[1, 1], [1, 1], [1, 1]])
        with self.assertRaises(ValueError):
            experiment.indicators_for_conditions(['unknown'], False)

    def test_independent_imports_and_balanced_split(self):
        tree = ast.parse(Path(experiment.__file__).read_text())
        imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        imports += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
        self.assertFalse(any(name and name.startswith(('run_ravdess_interaction', 'run_ravdess_fusion', 'run_ravdess_dbm'))
                             for name in imports))
        rows = [dict(video=f'{actor}-{label}-{trial}', audio='audio', actor=actor, label=label)
                for actor in range(1, 25) for label in range(6) for trial in range(8)]
        splits, report = experiment.balanced_split(rows, 42)
        self.assertEqual([len(splits[name]) for name in ('train', 'test')], [768, 384])
        self.assertFalse({r['video'] for r in splits['train']} & {r['video'] for r in splits['test']})
        expanded = experiment.expand_training_rows(splits['train'])
        self.assertEqual(len(expanded), 2304)
        for condition in experiment.TEST_CONDITIONS:
            self.assertEqual([sum(r['condition']==condition and r['label']==label for r in expanded) for label in range(6)], [128]*6)

    def test_prepare_runs_with_only_the_standalone_script(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / 'standalone.py'
            script.write_text(Path(experiment.__file__).read_text())
            video, audio = root / 'videos', root / 'audio'
            video.mkdir(); audio.mkdir()
            for actor in range(1, 25):
                for code in (1, 3, 4, 5, 6, 7, 8):
                    for intensity in ([1] if code==1 else [1, 2]):
                        for statement in (1, 2):
                            for repetition in (1, 2):
                                stem = f'01-01-{code:02d}-{intensity:02d}-{statement:02d}-{repetition:02d}-{actor:02d}'
                                (video / (stem+'.mp4')).touch()
                                (audio / (stem+'.wav')).touch()
            output = root / 'run'
            result = subprocess.run([sys.executable, str(script), '--video-root', str(video), '--audio-root', str(audio),
                                     '--output-dir', str(output), '--prepare-only'], cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            config = json.loads((output/'config.json').read_text())
            self.assertEqual(config['architecture'], 'indicator_mlp')
            self.assertEqual(config['initialization'], 'random')
            self.assertEqual(config['test_indicators'], [1, 1])
            self.assertEqual(config['neutral_frame_policy'], 'first')
            with (output/'train_expanded.csv').open() as f:
                expanded = list(csv.DictReader(f))
            self.assertEqual(len(expanded), 2304)
            for row in expanded:
                expected = experiment.indicators_for_conditions([row['condition']], True)[0]
                self.assertEqual([int(row['on_v']), int(row['on_a'])], expected)
            for option in (['--head-dim', '0'], ['--lr', 'nan'], ['--mlp-dropout', '1']):
                failed = subprocess.run([sys.executable, str(script), '--video-root', str(video), '--audio-root', str(audio),
                                         '--prepare-only', *option], cwd=root, capture_output=True, text=True)
                self.assertEqual(failed.returncode, 2)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Requires PyTorch')
class IndicatorModelTests(unittest.TestCase):
    def test_concatenation_gradients_default_test_bits_and_checkpoint(self):
        import torch
        nn = torch.nn
        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(1, 3, 1, 384))
            def forward(self, x):
                return self.weight.expand(len(x), -1, -1, -1)
        def backbone(**kwargs):
            self.assertEqual(kwargs['fusion_depth'], 0)
            return SimpleNamespace(encoder=Encoder(), encoder_audio=Encoder(),
                                   video_layer_weights=nn.Parameter(torch.ones(1)), audio_layer_weights=nn.Parameter(torch.ones(1)))
        args = SimpleNamespace(embed_dim=384, depth=1, drop_path=0., head_dim=8, mlp_hidden_dims=[12, 8], mlp_dropout=.1)
        fake = SimpleNamespace(AudioVisionTransformer=backbone)
        with patch.dict('sys.modules', {'modeling_finetune_av': fake}):
            model = experiment.make_model(args).eval()
            restored = experiment.make_model(args).eval()
        self.assertFalse(hasattr(model, 'encoder_fusion'))
        seen = []
        hook = model.classifier[0].register_forward_pre_hook(lambda module, inputs: seen.append(inputs[0].detach().clone()))
        video, audio = torch.ones(3, 1), torch.ones(3, 1)
        bits = torch.tensor([[1., 1.], [1., 0.], [0., 1.]])
        logits, terms = model(video, audio, indicators=bits, return_terms=True)
        torch.testing.assert_close(logits, sum(terms))
        self.assertEqual(seen[-1].shape, (3, 18))
        torch.testing.assert_close(seen[-1][:, -2:], bits)
        hv = model.video_head(model.encoder.weight.mean(1).squeeze(1)).expand(3, -1)
        ha = model.audio_head(model.encoder_audio.weight.mean(1).squeeze(1)).expand(3, -1)
        torch.testing.assert_close(seen[-1][:, :-2], torch.cat((hv, ha), -1))
        torch.testing.assert_close(model(video, audio), model(video, audio, indicators=torch.ones(3, 2)))
        torch.testing.assert_close(seen[-1][:, -2:], torch.ones(3, 2))
        hook.remove()
        restored.load_state_dict(model.state_dict(), strict=True)
        torch.testing.assert_close(restored(video, audio), model(video, audio))
        model.train()
        with self.assertRaises(ValueError):
            model(video, audio)
        loss = nn.functional.cross_entropy(model(video, audio, indicators=bits), torch.tensor([0, 1, 2]))
        loss.backward()
        for encoder in (model.encoder, model.encoder_audio):
            self.assertGreater(encoder.weight.grad.abs().sum().item(), 0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'pretrained.pth'
            state = restored.state_dict()
            torch.save(dict(model=state), path)
            old_head = model.classifier[0].weight.detach().clone()
            experiment.load_pretrained(model, path)
            torch.testing.assert_close(model.classifier[0].weight, old_head)
            torch.save(dict(model={k: v for k, v in state.items() if not k.startswith('encoder_audio.')}), path)
            with self.assertRaises(ValueError):
                experiment.load_pretrained(model, path)


@unittest.skipUnless(all(importlib.util.find_spec(m) for m in ('torch', 'numpy', 'sklearn')),
                     'Requires PyTorch/numpy/sklearn')
class IndicatorTrainerTests(unittest.TestCase):
    def test_mixed_training_bits_and_forced_test_bits_in_saved_outputs(self):
        import io
        from contextlib import redirect_stdout
        import torch
        nn = torch.nn
        class Dataset(torch.utils.data.Dataset):
            def __init__(self, condition='both', training=False):
                self.condition = condition
                self.rows = [dict(video=str(label), audio='x', actor=1, label=label) for label in range(6)]
                if training:
                    self.rows = experiment.expand_training_rows(self.rows)
            def __len__(self):
                return len(self.rows)
            def __getitem__(self, index):
                row = self.rows[index]
                return torch.eye(6)[row['label']], torch.eye(6)[row['label']], row['label'], index
        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.classifier = nn.Linear(14, 6)
            def forward(self, video, audio, indicators=None, return_terms=False):
                logits = self.classifier(torch.cat((video, audio, indicators), -1))
                return (logits, (logits,)) if return_terms else logits
        loaders = {'train': torch.utils.data.DataLoader(Dataset(training=True), batch_size=4, shuffle=True)}
        for condition in experiment.TEST_CONDITIONS:
            loaders['test_'+condition] = torch.utils.data.DataLoader(Dataset(condition), batch_size=4)
        args = SimpleNamespace(seed=42, device='cpu', checkpoint=None, lr=.001, weight_decay=.05, epochs=2,
                               skip_epoch_tests=False, skip_explanations=True)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch.object(experiment, 'make_model', return_value=Model()), \
                 patch.object(experiment, 'make_loaders', return_value=loaders), \
                 patch.object(experiment, 'load_pretrained') as pretrain, redirect_stdout(io.StringIO()):
                experiment.train({'train': [], 'test': []}, args, output)
            pretrain.assert_not_called()
            with (output/'epoch_predictions'/'train_epoch_001.csv').open() as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 18)
            for row in rows:
                self.assertEqual([int(row['on_v']), int(row['on_a'])], experiment.indicators_for_conditions([row['condition']], True)[0])
            for condition in experiment.TEST_CONDITIONS:
                with (output/('test_predictions_'+condition+'.csv')).open() as f:
                    rows = list(csv.DictReader(f))
                self.assertEqual({(row['on_v'], row['on_a']) for row in rows}, {('1', '1')})
                for row in rows:
                    self.assertEqual([int(row['available_v']), int(row['available_a'])],
                                     experiment.indicators_for_conditions([condition], True)[0])
                    for emotion in experiment.EMOTIONS:
                        self.assertEqual(row['logit_'+emotion], row['mlp_logit_'+emotion])
            summary = json.loads((output/'test_metrics.json').read_text())
            self.assertEqual(summary['architecture'], 'indicator_mlp')
            self.assertEqual(summary['test_indicators'], [1, 1])
            self.assertEqual(summary['initialization'], 'random')
            self.assertTrue((output/'final.pth').exists())
            self.assertFalse((output/'best.pth').exists())


@unittest.skipUnless(all(importlib.util.find_spec(m) for m in ('numpy', 'matplotlib')),
                     'Requires numpy/matplotlib')
class IndicatorAnalysisTests(unittest.TestCase):
    def test_joint_mlp_output_analysis(self):
        from analyze_ravdess_results import EMOTIONS, CONDITIONS, write_csv, recorded_branches, analyze_modality_contributions
        rows = []
        for label in range(6):
            row = dict(video=str(label), label=label, prediction=label, on_v=1, on_a=1)
            for index, emotion in enumerate(EMOTIONS):
                row['logit_'+emotion] = row['mlp_logit_'+emotion] = float(index==label)
            rows.append(row)
        self.assertEqual(recorded_branches(rows), ('mlp',))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for condition in CONDITIONS:
                write_csv(root/('test_predictions_'+condition+'.csv'), rows, list(rows[0]))
            self.assertTrue(analyze_modality_contributions(root, root, 40, []))


if __name__ == '__main__':
    unittest.main()
