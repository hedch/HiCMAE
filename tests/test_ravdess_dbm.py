import argparse
import importlib.util
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from run_ravdess_dbm import configure_parser, validate_args, CONDITION_CLASSES
from run_ravdess_interaction import condition_ids, expand_training_rows


class DBMConfigurationTests(unittest.TestCase):
    def test_pairwise_targets_masks_and_weights(self):
        from run_ravdess_dbm import pairwise_targets, pairwise_lambdas
        self.assertEqual(pairwise_targets(['both', 'video_noise', 'audio_neutral']),
                         {'audio_vs_av': [0, None, 1], 'video_vs_av': [0, 1, None]})
        self.assertEqual(pairwise_targets(['both', 'video_noise', 'audio_neutral'],
                                         {'audio_vs_video': ('audio_neutral', 'video_noise')}),
                         {'audio_vs_video': [None, 1, 0]})
        args = SimpleNamespace(condition_lambda=.1, video_av_lambda=.3)
        self.assertEqual(pairwise_lambdas(args), {'audio_vs_av': .1, 'video_vs_av': .3})
        for value in (-1., float('nan'), float('inf')):
            args.audio_av_lambda = value
            with self.assertRaises(ValueError):
                pairwise_lambdas(args)
        with self.assertRaises(ValueError):
            pairwise_targets(['unknown'])

    def test_defaults_and_invalid_parameters(self):
        parser = argparse.ArgumentParser()
        configure_parser(parser)
        args = parser.parse_args([])
        validate_args(args)
        self.assertEqual(args.dbm_hidden_dims, [256, 128])
        self.assertEqual(args.condition_adversary_mode, 'audio_vs_video')
        self.assertEqual(args.condition_classes, ['both', 'video_noise', 'audio_neutral'])
        for key, bad in (('condition_lambda', -1.), ('condition_lambda', float('nan')),
                         ('condition_lambda', float('inf')), ('dbm_steps', 0),
                         ('dbm_visible_dim', 0), ('dbm_hidden_dims', [0, 3])):
            invalid = SimpleNamespace(**vars(args))
            setattr(invalid, key, bad)
            with self.assertRaises(ValueError):
                validate_args(invalid)

    def test_condition_labels_match_actual_input_construction(self):
        rows = expand_training_rows([dict(video=str(i), actor=1, label=i) for i in range(6)])
        labels = condition_ids(rows, range(len(rows)))
        self.assertEqual([labels.count(i) for i in range(3)], [6, 6, 6])
        self.assertEqual(condition_ids([dict(label=5)], [0], 'audio_neutral'), [2])
        for row, label in zip(rows, labels):
            self.assertEqual(CONDITION_CLASSES[label], row['condition'])
        with self.assertRaises(ValueError):
            condition_ids([dict(condition='invalid')], [0])


@unittest.skipUnless(importlib.util.find_spec('torch'), 'Requires PyTorch')
class DBMTorchTests(unittest.TestCase):
    def test_binary_losses_exclude_unrelated_rows(self):
        import torch
        from run_ravdess_dbm import pairwise_cross_entropies
        scores = {name: torch.randn(3, 2, requires_grad=True) for name in ('audio_vs_av', 'video_vs_av')}
        pairs = pairwise_cross_entropies(scores, ['both', 'video_noise', 'audio_neutral'])
        loss = sum(pair['losses'].mean() for pair in pairs.values())
        loss.backward()
        self.assertEqual(scores['audio_vs_av'].grad[1].abs().sum().item(), 0)
        self.assertEqual(scores['video_vs_av'].grad[2].abs().sum().item(), 0)
        self.assertGreater(scores['audio_vs_av'].grad[[0, 2]].abs().sum().item(), 0)
        empty = pairwise_cross_entropies({'audio_vs_av': torch.ones(2, 2)}, ['video_noise', 'video_noise'])
        self.assertEqual(len(empty['audio_vs_av']['losses']), 0)
        scores = torch.randn(3, 2, requires_grad=True)
        single = pairwise_cross_entropies({'audio_vs_video': scores}, ['both', 'video_noise', 'audio_neutral'])
        single['audio_vs_video']['losses'].mean().backward()
        self.assertEqual(scores.grad[0].abs().sum().item(), 0)
        self.assertGreater(scores.grad[1:].abs().sum().item(), 0)
        empty = pairwise_cross_entropies({'audio_vs_video': torch.ones(2, 2)}, ['both', 'both'])
        self.assertEqual(len(empty['audio_vs_video']['losses']), 0)

    def test_gradient_reversal_changes_features_but_not_classifier(self):
        import torch
        from dbm_fusion import reverse_gradient
        torch.manual_seed(4)
        classifier = torch.nn.Linear(4, 3)
        features = torch.randn(5, 4, requires_grad=True)
        target = torch.tensor([0, 1, 2, 0, 1])
        ordinary = torch.nn.functional.cross_entropy(classifier(features), target)
        expected = torch.autograd.grad(ordinary, (features, classifier.weight, classifier.bias))
        for coefficient in (0., .3, 1.):
            adversarial = torch.nn.functional.cross_entropy(classifier(reverse_gradient(features, coefficient)), target)
            gradients = torch.autograd.grad(adversarial, (features, classifier.weight, classifier.bias))
            torch.testing.assert_close(gradients[0], -coefficient * expected[0])
            torch.testing.assert_close(gradients[1], expected[1])
            torch.testing.assert_close(gradients[2], expected[2])

    def test_dbm_energy_mean_field_and_gradients(self):
        import torch
        from dbm_fusion import MultimodalDBM
        dbm = MultimodalDBM(3, (4, 2), steps=3)
        v, a = torch.randn(2, 3, requires_grad=True), torch.randn(2, 3, requires_grad=True)
        drive = v @ dbm.w_video + a @ dbm.w_audio + dbm.b1
        h1 = drive.sigmoid()
        h2 = (h1 @ dbm.w_hidden + dbm.b2).sigmoid()
        for _ in range(3):
            h1 = (drive + h2 @ dbm.w_hidden.T).sigmoid()
            h2 = (h1 @ dbm.w_hidden + dbm.b2).sigmoid()
        actual1, actual2 = dbm(v, a, return_states=True)
        torch.testing.assert_close(actual1, h1)
        torch.testing.assert_close(actual2, h2)
        # Conditional Bernoulli log-odds from energy differences use the same W.
        zeros = torch.zeros(2, 4)
        ones = zeros.clone(); ones[:, 0] = 1
        log_odds = -(dbm.energy(v, a, ones, h2) - dbm.energy(v, a, zeros, h2))
        torch.testing.assert_close(log_odds, (drive + h2 @ dbm.w_hidden.T)[:, 0], atol=1e-6, rtol=1e-5)
        actual2.square().sum().backward()
        for value in [v, a, *dbm.parameters()]:
            self.assertIsNotNone(value.grad)
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(value.grad.abs().sum().item(), 0)
        with self.assertRaises(ValueError):
            MultimodalDBM(3, (4, 2), steps=0)

    def test_full_model_checkpoint_and_separate_condition_head(self):
        import torch
        from run_ravdess_dbm import make_model, load_pretrained
        nn = torch.nn
        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(1, 3, 1, 384))
            def forward(self, x):
                return self.weight.expand(len(x), -1, -1, -1)
        def backbone(**kwargs):
            return SimpleNamespace(encoder=Encoder(), encoder_audio=Encoder(),
                                   video_layer_weights=nn.Parameter(torch.ones(1)),
                                   audio_layer_weights=nn.Parameter(torch.ones(1)))
        args = SimpleNamespace(embed_dim=384, depth=1, drop_path=0., dbm_visible_dim=8,
                               dbm_hidden_dims=[12, 6], dbm_steps=2,
                               condition_hidden_dim=8, condition_lambda=.3)
        fake_module = SimpleNamespace(AudioVisionTransformer=backbone)
        with patch.dict('sys.modules', {'modeling_finetune_av': fake_module}):
            model = make_model(args).eval()
            restored = make_model(args).eval()
        self.assertFalse(hasattr(model, 'encoder_fusion'))
        self.assertEqual(model.logit_terms, ('dbm',))
        v, a = torch.ones(3, 1), torch.ones(3, 1)
        expected = model(v, a)
        logits, terms, conditions = model(v, a, return_adversary=True)
        torch.testing.assert_close(expected, logits)
        torch.testing.assert_close(logits, sum(terms))
        restored.load_state_dict(model.state_dict(), strict=True)
        torch.testing.assert_close(restored(v, a), logits)
        with torch.no_grad():
            for parameter in model.condition_classifier.parameters():
                parameter.add_(100)
        torch.testing.assert_close(model(v, a), logits)
        # Main-only loss never trains the condition classifier.
        model(v, a).sum().backward()
        self.assertTrue(all(p.grad is None for p in model.condition_classifier.parameters()))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'checkpoint.pth'
            state = restored.state_dict()
            torch.save(dict(model=state), path)
            old_head = model.heads['dbm'].weight.detach().clone()
            load_pretrained(model, path)
            torch.testing.assert_close(model.heads['dbm'].weight, old_head)
            torch.testing.assert_close(model.encoder.weight, restored.encoder.weight)
            torch.save(dict(model={k: v for k, v in state.items() if not k.startswith('encoder_audio.')}), path)
            with self.assertRaises(ValueError):
                load_pretrained(model, path)
        args.condition_adversary_mode = 'pairwise'
        args.audio_av_lambda, args.video_av_lambda = .2, .5
        with patch.dict('sys.modules', {'modeling_finetune_av': fake_module}):
            paired = make_model(args).eval()
            paired_restore = make_model(args).eval()
        logits, _, discriminators = paired(v, a, return_adversary=True)
        self.assertEqual(set(discriminators), {'audio_vs_av', 'video_vs_av'})
        self.assertTrue(all(scores.shape == (3, 2) for scores in discriminators.values()))
        self.assertEqual(paired.condition_lambdas, {'audio_vs_av': .2, 'video_vs_av': .5})
        paired_restore.load_state_dict(paired.state_dict(), strict=True)
        torch.testing.assert_close(logits, paired_restore(v, a))
        paired(v, a).sum().backward()
        self.assertTrue(all(parameter.grad is None for parameter in paired.condition_classifiers.parameters()))
        args.condition_adversary_mode = 'audio_vs_video'
        with patch.dict('sys.modules', {'modeling_finetune_av': fake_module}):
            single = make_model(args).eval()
            single_restore = make_model(args).eval()
        self.assertEqual(tuple(single.condition_classifiers), ('audio_vs_video',))
        logits, _, adversaries = single(v, a, return_adversary=True)
        self.assertEqual(tuple(adversaries), ('audio_vs_video',))
        single_restore.load_state_dict(single.state_dict(), strict=True)
        torch.testing.assert_close(single_restore(v, a), logits)


@unittest.skipUnless(all(importlib.util.find_spec(m) for m in ('numpy', 'matplotlib')),
                     'Requires numpy/matplotlib')
class DBMAnalysisTests(unittest.TestCase):
    def test_dbm_scores_and_condition_learning_figures(self):
        import numpy as np
        from analyze_ravdess_results import (EMOTIONS, CONDITIONS, write_csv, read_csv,
            contribution_arrays, recorded_branches, analyze_modality_contributions, plot_condition_learning)
        rows = []
        for label in range(6):
            row = dict(video=str(label), label=label, prediction=label)
            for i, emotion in enumerate(EMOTIONS):
                row['logit_' + emotion] = row['dbm_logit_' + emotion] = float(i == label)
            rows.append(row)
        self.assertEqual(recorded_branches(rows), ('dbm',))
        scores, terms = contribution_arrays(rows)
        np.testing.assert_allclose(scores, terms[:, 0])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for condition in CONDITIONS:
                write_csv(root / ('test_predictions_' + condition + '.csv'), rows, list(rows[0]))
            self.assertTrue(analyze_modality_contributions(root, root, 40, []))
            ablations = read_csv(root / 'modality_contributions' / 'term_ablation_accuracy.csv')
            self.assertFalse(any(row['configuration'].startswith('drop_') for row in ablations))
            history = [dict(epoch=epoch, split=split, total=18 if split == 'train' else 6,
                            accuracy=1/3, loss=np.log(3), entropy=np.log(3))
                       for epoch in (1, 2) for split in ('train', 'test_both', 'test_video_noise', 'test_audio_neutral')]
            write_csv(root / 'condition_metrics.csv', history, list(history[0]))
            self.assertTrue(plot_condition_learning(root, root, 40))
            self.assertTrue((root / 'condition_adversary_learning.png').exists())
            self.assertTrue((root / 'condition_adversary_learning.pdf').exists())
            pair_history = [dict(epoch=epoch, split=split, adversary=name, total=12 if split == 'train' else 6,
                                 accuracy=.5, loss=np.log(2), entropy=np.log(2))
                            for epoch in (1, 2) for name, control in (('audio_vs_av', 'audio_neutral'), ('video_vs_av', 'video_noise'))
                            for split in ('train', 'test_both', 'test_' + control)]
            write_csv(root / 'pairwise_condition_metrics.csv', pair_history, list(pair_history[0]))
            self.assertTrue(plot_condition_learning(root, root, 40))
            for name in ('audio_vs_av', 'video_vs_av'):
                self.assertTrue((root / ('condition_adversary_learning_' + name + '.png')).exists())
            single_history = [dict(epoch=epoch, split=split, adversary='audio_vs_video', total=12 if split == 'train' else 6,
                                   accuracy=.5, loss=np.log(2), entropy=np.log(2))
                              for epoch in (1, 2) for split in ('train', 'test_audio_neutral', 'test_video_noise')]
            write_csv(root / 'pairwise_condition_metrics.csv', single_history, list(single_history[0]))
            self.assertTrue(plot_condition_learning(root, root, 40))
            self.assertTrue((root / 'condition_adversary_learning_audio_vs_video.png').exists())


@unittest.skipUnless(all(importlib.util.find_spec(m) for m in ('torch', 'numpy', 'sklearn')),
                     'Requires PyTorch/numpy/sklearn')
class DBMTrainerTests(unittest.TestCase):
    def test_adversarial_training_labels_and_saved_outputs(self):
        self.check_training_outputs(False)

    def test_pairwise_training_labels_and_saved_outputs(self):
        self.check_training_outputs(True)

    def test_single_audio_vs_video_training(self):
        self.check_training_outputs('audio_vs_video')

    def check_training_outputs(self, paired):
        import csv
        import json
        import io
        from contextlib import redirect_stdout
        import torch
        import run_ravdess_interaction as experiment
        from dbm_fusion import reverse_gradient
        nn = torch.nn
        class Dataset(torch.utils.data.Dataset):
            def __init__(self, condition='both', expanded=False):
                self.condition = condition
                self.rows = [dict(video=str(label), actor=1, label=label) for label in range(6)]
                if expanded:
                    self.rows = expand_training_rows(self.rows)
            def __len__(self):
                return len(self.rows)
            def __getitem__(self, index):
                row = self.rows[index]
                condition = row.get('condition', self.condition)
                real = torch.eye(6)[row['label']]
                return (real if condition != 'audio_neutral' else torch.zeros(6),
                        real if condition != 'video_noise' else torch.zeros(6), row['label'], index)
        class Model(nn.Module):
            supports_condition_adversary = True
            condition_lambda = .4
            logit_terms = ('dbm',)
            def __init__(self):
                super().__init__()
                self.features = nn.Linear(12, 6)
                self.heads = nn.ModuleDict({'dbm': nn.Linear(6, 6, bias=False)})
                if paired:
                    self.condition_lambdas = ({'audio_vs_video': .4} if paired == 'audio_vs_video'
                                              else {'audio_vs_av': .2, 'video_vs_av': .5})
                    self.condition_classifiers = nn.ModuleDict({name: nn.Linear(6, 2) for name in self.condition_lambdas})
                else:
                    self.condition_classifier = nn.Linear(6, 3)
            def forward(self, v, a, return_terms=False, return_adversary=False):
                features = self.features(torch.cat((v, a), -1)).tanh()
                logits = self.heads['dbm'](features)
                if return_adversary:
                    if paired:
                        return logits, (logits,), {name: head(reverse_gradient(features, self.condition_lambdas[name]))
                                                   for name, head in self.condition_classifiers.items()}
                    return logits, (logits,), self.condition_classifier(reverse_gradient(features, self.condition_lambda))
                return (logits, (logits,)) if return_terms else logits
        loaders = {'train': torch.utils.data.DataLoader(Dataset(expanded=True), batch_size=4, shuffle=True)}
        for condition in CONDITION_CLASSES:
            loaders['test_' + condition] = torch.utils.data.DataLoader(Dataset(condition), batch_size=4)
        args = SimpleNamespace(seed=42, device='cpu', checkpoint='unused', lr=.001, epochs=2,
                               architecture='dbm', skip_epoch_tests=False, skip_explanations=True)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            with patch.object(experiment, 'make_loaders', return_value=loaders), redirect_stdout(io.StringIO()):
                experiment.train({'train': [], 'test': []}, args, output,
                                 model_factory=lambda args: Model(), pretrained_loader=lambda model, path: None)
            with (output / ('pairwise_condition_metrics.csv' if paired else 'condition_metrics.csv')).open() as f:
                metrics = list(csv.DictReader(f))
            self.assertEqual(len(metrics), 6 if paired == 'audio_vs_video' else 12 if paired else 8)
            if paired == 'audio_vs_video':
                self.assertNotIn('test_both', {row['split'] for row in metrics})
            self.assertFalse((output / 'auxiliary_metrics.csv').exists())
            history = [json.loads(line) for line in (output / 'history.jsonl').read_text().splitlines()]
            for entry in history:
                if paired:
                    adv = entry['train']['pairwise_condition_adversaries']
                    self.assertTrue(all(stats['total'] == 12 for stats in adv.values()))
                    self.assertAlmostEqual(entry['train']['adversarial_feature_objective'],
                                           entry['train']['loss'] - sum(stats['coefficient']*stats['loss'] for stats in adv.values()), places=6)
                else:
                    adv = entry['train']['condition_adversary']
                    self.assertEqual(adv['total'], 18)
                    self.assertAlmostEqual(adv['feature_objective'], entry['train']['loss'] - .4*adv['loss'], places=6)
                    self.assertAlmostEqual(adv['backward_surrogate'], entry['train']['loss'] + adv['loss'], places=6)
            for index, condition in enumerate(CONDITION_CLASSES):
                with (output / ('test_predictions_' + condition + '.csv')).open() as f:
                    predictions = list(csv.DictReader(f))
                if paired:
                    pairs = ({'audio_vs_video': ('audio_neutral', 'video_noise')} if paired == 'audio_vs_video'
                             else {'audio_vs_av': ('both', 'audio_neutral'), 'video_vs_av': ('both', 'video_noise')})
                    for name, pair in pairs.items():
                        expected = '' if condition not in pair else str(pair.index(condition))
                        self.assertEqual({row[name + '_target'] for row in predictions}, {expected})
                else:
                    self.assertEqual({int(row['condition_target']) for row in predictions}, {index})
                for row in predictions:
                    for emotion in experiment.EMOTIONS:
                        self.assertEqual(row['logit_' + emotion], row['dbm_logit_' + emotion])
            checkpoint = torch.load(output / 'final.pth', map_location='cpu')
            model = Model()
            model.load_state_dict(checkpoint['model'], strict=True)


if __name__ == '__main__':
    unittest.main()
