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
    def test_fixed_first_layer_partition_dimensions(self):
        self.assertEqual(experiment.first_layer_half_width(256), 128)
        sizes = experiment.first_layer_gate_sizes
        self.assertEqual(sizes(256), (128, 128))
        self.assertEqual(sizes(256, 192, 64), (192, 64))
        self.assertEqual(sizes(256, 32, 64), (32, 64))
        self.assertEqual(sizes(256, visual=160), (160, 96))
        self.assertEqual(sizes(256, audio=160), (96, 160))
        self.assertEqual(sizes(256, 5, 8), (5, 8))
        for visual, audio in ((0, 128), (-1, 128), (256, None), (None, 256)):
            with self.assertRaises(ValueError):
                sizes(256, visual, audio)
        for width in (0, 1, 3, 255):
            with self.assertRaises(ValueError):
                experiment.first_layer_half_width(width)

    def test_gpu_selection_and_validation(self):
        resolve = experiment.resolve_gpu_selection
        self.assertEqual(resolve('cuda', [0, 1, 2, 3], 4), ('cuda:0', [0, 1, 2, 3]))
        self.assertEqual(resolve('cuda', [2], 4), ('cuda:2', [2]))
        self.assertEqual(resolve('cuda:3', None, 4), ('cuda:3', [3]))
        self.assertEqual(resolve('cuda:3', [2, 0], 4), ('cuda:2', [2, 0]))
        self.assertEqual(resolve('cpu', None, 0), ('cpu', []))
        for device, ids, count in (('cuda', [0, 0], 4), ('cuda', [-1], 4), ('cuda', [4], 4),
                                    ('cuda', [0], 0), ('cpu', [0], 4), ('cuda:bad', None, 4)):
            with self.assertRaises(ValueError):
                resolve(device, ids, count)

    def test_actor_split_incomplete_data_and_rejections(self):
        rows = [dict(video=f'{actor}-{label}-{trial}', audio='audio', actor=actor, label=label)
                for actor in range(1, 25) for label in range(6) for trial in range(8)]
        _, baseline = experiment.balanced_split(rows, 42)
        held_out = baseline['test']['actors'][0]
        incomplete = [row for row in rows if row['video'] != f'{held_out}-0-0']
        splits, report = experiment.balanced_split(incomplete, 42)
        self.assertEqual(report['train']['balanced_counts'], [160]*6)
        self.assertEqual(report['test']['balanced_counts'], [31]*6)
        self.assertEqual(report['test']['unused_counts'], [0, 1, 1, 1, 1, 1])
        self.assertFalse({row['actor'] for row in splits['train']} & {row['actor'] for row in splits['test']})
        for invalid in (rows + [rows[0]], [row for row in rows if row['actor'] != 24],
                        [row for row in rows if row['label'] != 0]):
            with self.assertRaises(ValueError):
                experiment.balanced_split(invalid, 42)

    def test_training_and_testing_indicators(self):
        self.assertEqual(experiment.indicators_for_conditions(experiment.TEST_CONDITIONS, True),
                         [[1, 1], [1, 0], [0, 1]])
        self.assertEqual(experiment.indicators_for_conditions(experiment.TEST_CONDITIONS, False),
                         [[1, 1], [1, 0], [1, 1]])
        self.assertEqual(experiment.indicators_for_conditions(experiment.TEST_CONDITIONS, False, [1, 1]),
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
        self.assertEqual([len(splits[name]) for name in ('train', 'test')], [960, 192])
        self.assertFalse({r['video'] for r in splits['train']} & {r['video'] for r in splits['test']})
        self.assertFalse({r['actor'] for r in splits['train']} & {r['actor'] for r in splits['test']})
        self.assertEqual([len(report[name]['actors']) for name in ('train', 'test')], [20, 4])
        self.assertEqual((splits, report), experiment.balanced_split(rows, 42))
        self.assertNotEqual(report['test']['actors'], experiment.balanced_split(rows, 43)[1]['test']['actors'])
        self.assertEqual(report['test']['balanced_counts'], [32]*6)
        expanded = experiment.expand_training_rows(splits['train'])
        self.assertEqual(len(expanded), 2880)
        for condition in experiment.TEST_CONDITIONS:
            self.assertEqual([sum(r['condition']==condition and r['label']==label for r in expanded) for label in range(6)], [160]*6)

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
                                     '--output-dir', str(output), '--prepare-only', '--gpu-ids', '0', '1', '2', '3',
                                     '--batch-size', '16'], cwd=root, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            config = json.loads((output/'config.json').read_text())
            self.assertEqual(config['architecture'], 'indicator_mlp')
            self.assertEqual(config['gpu_ids'], [0, 1, 2, 3])
            self.assertEqual(config['device'], 'cuda:0')
            self.assertEqual(config['initialization'], 'random')
            self.assertEqual(config['split_method'], 'actor_disjoint_20_4')
            self.assertEqual(len(config['train_actors']), 20)
            self.assertEqual(len(config['test_actors']), 4)
            self.assertFalse(set(config['train_actors']) & set(config['test_actors']))
            for name in ('train', 'test'):
                with (output/(name+'.csv')).open() as f:
                    manifest = list(csv.DictReader(f))
                self.assertEqual({int(row['actor']) for row in manifest}, set(config[name+'_actors']))
                for row in manifest:
                    self.assertEqual(int(Path(row['neutral_video']).stem.split('-')[-1]), int(row['actor']))
            self.assertIsNone(config['test_indicators'])
            self.assertEqual(config['mlp_input_mode'], 'gradient_gated')
            self.assertTrue(config['first_layer_gate_assignment']['fixed'])
            self.assertEqual([config['visual_gated_neurons'], config['audio_gated_neurons']], [128, 128])
            self.assertEqual(config['neutral_frame_policy'], 'first')
            with (output/'train_expanded.csv').open() as f:
                expanded = list(csv.DictReader(f))
            self.assertEqual(len(expanded), 2880)
            for row in expanded:
                expected = experiment.indicators_for_conditions([row['condition']], True)[0]
                self.assertEqual([int(row['on_v']), int(row['on_a'])], expected)
            for option in (['--head-dim', '0'], ['--lr', 'nan'], ['--mlp-dropout', '1'],
                           ['--gpu-ids', '0', '0'], ['--device', 'cpu', '--gpu-ids', '1'],
                           ['--mlp-hidden-dims', '255', '128'], ['--visual-gated-neurons', '0'],
                           ['--audio-gated-neurons', '-1'], ['--visual-gated-neurons', '256']):
                failed = subprocess.run([sys.executable, str(script), '--video-root', str(video), '--audio-root', str(audio),
                                         '--prepare-only', *option], cwd=root, capture_output=True, text=True)
                self.assertEqual(failed.returncode, 2)
            for i, (options, expected) in enumerate((
                    (['--visual-gated-neurons', '192', '--audio-gated-neurons', '64'], [192, 64]),
                    (['--visual-gated-neurons', '32', '--audio-gated-neurons', '64'], [32, 64]),
                    (['--visual-gated-neurons', '160'], [160, 96]),
                    (['--audio-gated-neurons', '160'], [96, 160]),
                    (['--visual-gated-neurons', '5', '--audio-gated-neurons', '8'], [5, 8]))):
                custom = root / ('custom%d' % i)
                result = subprocess.run([sys.executable, str(script), '--video-root', str(video), '--audio-root', str(audio),
                                         '--output-dir', str(custom), '--prepare-only', *options],
                                        cwd=root, capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                custom_config = json.loads((custom/'config.json').read_text())
                self.assertEqual([custom_config['visual_gated_neurons'], custom_config['audio_gated_neurons']], expected)
                self.assertEqual(custom_config['mlp_hidden_dims'], [sum(expected), 128])
                self.assertEqual(custom_config['first_layer_gate_assignment']['visual'], [0, expected[0]])
                self.assertEqual(custom_config['first_layer_gate_assignment']['audio'], [expected[0], sum(expected)])
                self.assertEqual((custom/'train.csv').read_text(), (output/'train.csv').read_text())
                self.assertEqual((custom/'test.csv').read_text(), (output/'test.csv').read_text())


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Requires PyTorch')
class IndicatorModelTests(unittest.TestCase):
    def test_fixed_backward_only_gates_and_adamw_inactive_half_freeze(self):
        import torch
        nn = torch.nn
        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(1, 3, 1, 384))
            def forward(self, x):
                return self.weight.expand(len(x), -1, -1, -1)
        def backbone(**kwargs):
            return SimpleNamespace(encoder=Encoder(), encoder_audio=Encoder(),
                                   video_layer_weights=nn.Parameter(torch.ones(1)), audio_layer_weights=nn.Parameter(torch.ones(1)))
        args = SimpleNamespace(embed_dim=384, depth=1, drop_path=0., head_dim=8,
                               mlp_hidden_dims=[12, 7, 9], mlp_dropout=0., mlp_input_mode='gradient_gated',
                               visual_gated_neurons=8, audio_gated_neurons=4)
        with patch.dict('sys.modules', {'modeling_finetune_av': SimpleNamespace(AudioVisionTransformer=backbone)}):
            model = experiment.make_model(args)
            restored = experiment.make_model(args).eval()
        first = model.classifier[0]
        self.assertEqual(first.visual.in_features, 16)
        self.assertEqual(first.visual.out_features, 8)
        self.assertEqual(first.audio.out_features, 4)
        self.assertFalse(hasattr(model, 'test_indicators'))
        self.assertIsNone(model.indicators_for_test_condition('video_noise'))
        parameter_ids = [id(first.visual.weight), id(first.audio.weight)]
        video, audio = torch.ones(3, 1), torch.ones(3, 1)
        bits = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])
        captures = []
        def capture(module, inputs, output):
            output.retain_grad()
            captures.append(output)
        handle = model.classifier[1].register_forward_hook(capture)
        logits = model(video, audio, indicators=bits)
        hidden = captures[-1]
        torch.testing.assert_close(logits, model(video, audio, indicators=torch.ones(3, 2)))
        logits.sum().backward()
        mask = torch.cat((bits[:, :1].expand(-1, 8), bits[:, 1:2].expand(-1, 4)), dim=-1).bool()
        self.assertEqual(hidden.grad[~mask].abs().sum().item(), 0)
        self.assertGreater(hidden.grad[mask].abs().sum().item(), 0)
        handle.remove()
        # Populate AdamW momentum before verifying that an inactive group is skipped.
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01, weight_decay=.2)
        def step(indicators):
            optimizer.zero_grad(set_to_none=True)
            loss = nn.functional.cross_entropy(model(video, audio, indicators=indicators), torch.tensor([0, 1, 2]))
            loss.backward()
            model.prepare_gradient_gate_step(indicators)
            optimizer.step()
        step(torch.ones(3, 2))
        for inactive, gate in ((first.visual, [0., 1.]), (first.audio, [1., 0.])):
            snapshots = {parameter: (parameter.detach().clone(),
                                     {key: value.clone() if torch.is_tensor(value) else value
                                      for key, value in optimizer.state[parameter].items()})
                         for parameter in inactive.parameters()}
            step(torch.tensor([gate]*3))
            for parameter, (before, state) in snapshots.items():
                self.assertIsNone(parameter.grad)
                torch.testing.assert_close(parameter, before, rtol=0, atol=0)
                for key, value in state.items():
                    if torch.is_tensor(value):
                        torch.testing.assert_close(optimizer.state[parameter][key], value, rtol=0, atol=0)
                    else:
                        self.assertEqual(optimizer.state[parameter][key], value)
            self.assertIsNotNone(model.classifier[-1].weight.grad)
        self.assertEqual(parameter_ids, [id(first.visual.weight), id(first.audio.weight)])
        model.eval()
        torch.testing.assert_close(model(video, audio), model(video, audio, indicators=torch.zeros(3, 2)))
        restored.load_state_dict(model.state_dict(), strict=True)
        torch.testing.assert_close(restored(video, audio), model(video, audio))

    def test_actual_data_parallel_when_multiple_gpus_available(self):
        import torch
        if torch.cuda.device_count() < 2:
            self.skipTest('Requires at least two CUDA GPUs')
        raw = torch.nn.Linear(4, 6)
        args = SimpleNamespace(device='cuda', gpu_ids=[0, 1], batch_size=8)
        wrapped, info = experiment.configure_parallel_model(raw, args)
        self.assertIsInstance(wrapped, torch.nn.DataParallel)
        self.assertIs(wrapped.module, raw)
        self.assertEqual(info['gpu_ids'], [0, 1])
        self.assertEqual(args.device, 'cuda:0')
        inputs = torch.randn(8, 4, device='cuda:0')
        torch.testing.assert_close(wrapped(inputs), raw(inputs))
        loss = torch.nn.functional.cross_entropy(wrapped(inputs), torch.arange(8, device='cuda:0') % 6)
        loss.backward()
        self.assertTrue(all(parameter.grad is not None for parameter in raw.parameters()))
        restored = torch.nn.Linear(4, 6)
        restored.load_state_dict(raw.state_dict(), strict=True)

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
        self.assertEqual(model.indicators_for_test_condition('video_noise'), [1, 1])
        model.test_indicators = experiment.TEST_INDICATORS
        self.assertEqual(model.indicators_for_test_condition('video_noise'), [1, 0])
        self.assertEqual(model.indicators_for_test_condition('audio_neutral'), [1, 1])
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
        self.check_training_outputs(False)

    def test_gradient_gated_training_and_indicator_free_testing(self):
        self.check_training_outputs(True)

    def check_training_outputs(self, gated):
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
                self.gradient_gating = gated
                self.classifier = nn.Linear(12 if gated else 14, 6)
            def prepare_gradient_gate_step(self, indicators):
                if indicators is None:
                    raise AssertionError('Training gate missing')
            def forward(self, video, audio, indicators=None, return_terms=False):
                if gated:
                    if self.training and indicators is None:
                        raise AssertionError('Training must supply gates')
                    if not self.training and indicators is not None:
                        raise AssertionError('Testing must not supply gates')
                    features = torch.cat((video, audio), -1)
                else:
                    features = torch.cat((video, audio, indicators), -1)
                logits = self.classifier(features)
                return (logits, (logits,)) if return_terms else logits
        loaders = {'train': torch.utils.data.DataLoader(Dataset(training=True), batch_size=4, shuffle=True)}
        for condition in experiment.TEST_CONDITIONS:
            loaders['test_'+condition] = torch.utils.data.DataLoader(Dataset(condition), batch_size=4)
        args = SimpleNamespace(seed=42, device='cpu', checkpoint=None, lr=.001, weight_decay=.05, epochs=2,
                               skip_epoch_tests=False, skip_explanations=True)
        if gated:
            args.mlp_input_mode, args.test_indicators = 'gradient_gated', None
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
                expected = ('', '') if gated else tuple(str(bit) for bit in experiment.TEST_INDICATORS[condition])
                self.assertEqual({(row['on_v'], row['on_a']) for row in rows}, {expected})
                for row in rows:
                    self.assertEqual([int(row['available_v']), int(row['available_a'])],
                                     experiment.indicators_for_conditions([condition], True)[0])
                    for emotion in experiment.EMOTIONS:
                        self.assertEqual(row['logit_'+emotion], row['mlp_logit_'+emotion])
            summary = json.loads((output/'test_metrics.json').read_text())
            self.assertEqual(summary['architecture'], 'indicator_mlp')
            self.assertEqual(summary['test_indicators'], None if gated else experiment.TEST_INDICATORS)
            self.assertEqual(summary['initialization'], 'random')
            self.assertTrue((output/'final.pth').exists())
            checkpoint = torch.load(output/'final.pth', map_location='cpu')
            self.assertFalse(any(key.startswith('module.') for key in checkpoint['model']))
            parallelism = json.loads((output/'parallelism.json').read_text())
            self.assertEqual(parallelism['mode'], 'single_device')
            self.assertEqual(parallelism['gpu_ids'], [])
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
