import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


@unittest.skipUnless(all(importlib.util.find_spec(m) for m in ('numpy', 'matplotlib')),
                     'Requires numpy and matplotlib')
class ModalityAnalysisTests(unittest.TestCase):
    def make_rows(self):
        import numpy as np
        from analyze_ravdess_results import EMOTIONS, BRANCHES
        rows = []
        for label in range(6):
            for i in range(2):
                terms = np.zeros((3, 6))
                terms[:, label] = [3, 2, 1]
                terms[:, (label+1) % 6] = [1 if i == 0 else 7, .5, .5]
                logits = terms.sum(0)
                row = dict(video='%d_%d.mp4' % (label, i), label=label,
                           prediction=int(logits.argmax()))
                row.update({'logit_' + e: logits[j] for j, e in enumerate(EMOTIONS)})
                for bi, branch in enumerate(('video', 'audio', 'interaction')):
                    row.update({branch + '_logit_' + e: terms[bi, j] for j, e in enumerate(EMOTIONS)})
                rows.append(row)
        return rows

    def test_reconstruction_and_corrupt_scores(self):
        from analyze_ravdess_results import contribution_arrays
        import numpy as np
        rows = self.make_rows()
        logits, terms = contribution_arrays(rows)
        np.testing.assert_allclose(terms.sum(1), logits)
        rows[0]['video_logit_happy'] += 10
        with self.assertRaises(ValueError):
            contribution_arrays(rows)

    def test_known_margin_and_paired_ablation_summaries(self):
        from analyze_ravdess_results import analyze_modality_contributions, CONDITIONS, write_csv, read_csv
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = self.make_rows()
            for condition in CONDITIONS:
                write_csv(root / ('test_predictions_' + condition + '.csv'), rows, list(rows[0]))
            warnings = []
            self.assertTrue(analyze_modality_contributions(root, root, 40, warnings))
            self.assertFalse(warnings)
            folder = root / 'modality_contributions'
            summary = read_csv(folder / 'contributions_by_emotion.csv')
            expected = dict(video=-1., audio=1.5, interaction=.5)
            self.assertEqual(len(summary), 54)
            for row in summary:
                self.assertAlmostEqual(float(row['mean_margin']), expected[row['branch']])
            for row in read_csv(folder / 'term_ablation_accuracy.csv'):
                if row['configuration'] == 'drop_video':
                    self.assertEqual(float(row['accuracy_drop']), -.5)
            self.assertEqual(len(list(folder.glob('*.png'))), 4)
            for condition in CONDITIONS:
                legacy = [{k: v for k, v in row.items() if 'logit_' not in k} for row in rows]
                write_csv(root / ('test_predictions_' + condition + '.csv'), legacy, list(legacy[0]))
            self.assertFalse(analyze_modality_contributions(root, root, 40, warnings))
            self.assertEqual(len(warnings), 1)

    def test_two_term_model_has_no_audio_contribution_or_ablation(self):
        from analyze_ravdess_results import (analyze_modality_contributions, CONDITIONS,
                                            EMOTIONS, write_csv, read_csv, contribution_arrays)
        rows = self.make_rows()
        for row in rows:
            for emotion in EMOTIONS:
                row['logit_' + emotion] -= row.pop('audio_logit_' + emotion)
        logits, terms = contribution_arrays(rows)
        self.assertEqual(terms.shape, (12, 2, 6))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for condition in CONDITIONS:
                write_csv(root / ('test_predictions_' + condition + '.csv'), rows, list(rows[0]))
            warnings = []
            self.assertTrue(analyze_modality_contributions(root, root, 40, warnings))
            folder = root / 'modality_contributions'
            summary = read_csv(folder / 'contributions_by_emotion.csv')
            self.assertEqual(len(summary), 36)
            self.assertEqual({row['branch'] for row in summary}, {'video', 'interaction'})
            for row in summary:
                self.assertAlmostEqual(float(row['mean_margin']), -1. if row['branch'] == 'video' else .5)
            ablations = read_csv(folder / 'term_ablation_accuracy.csv')
            self.assertEqual(len(ablations), 90)
            self.assertFalse(any('audio' in row['configuration'] for row in ablations))
            self.assertFalse(warnings)

    def test_video_fusion_results_use_fusion_labels(self):
        from analyze_ravdess_results import (analyze_modality_contributions, CONDITIONS,
                                            EMOTIONS, write_csv, read_csv, contribution_arrays)
        rows = self.make_rows()
        for row in rows:
            for emotion in EMOTIONS:
                row['logit_' + emotion] -= row.pop('audio_logit_' + emotion)
                row['fusion_logit_' + emotion] = row.pop('interaction_logit_' + emotion)
        _, terms = contribution_arrays(rows)
        self.assertEqual(terms.shape, (12, 2, 6))
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for condition in CONDITIONS:
                write_csv(root / ('test_predictions_' + condition + '.csv'), rows, list(rows[0]))
            warnings = []
            self.assertTrue(analyze_modality_contributions(root, root, 40, warnings))
            folder = root / 'modality_contributions'
            summary = read_csv(folder / 'contributions_by_emotion.csv')
            self.assertEqual({row['branch'] for row in summary}, {'video', 'fusion'})
            ablations = read_csv(folder / 'term_ablation_accuracy.csv')
            self.assertEqual({row['configuration'] for row in ablations},
                             {'full', 'drop_video', 'drop_fusion', 'video_term_only', 'fusion_term_only'})
            self.assertFalse(warnings)

    def test_selectable_four_heads_and_single_head_analysis(self):
        import itertools
        import numpy as np
        from analyze_ravdess_results import (analyze_modality_contributions, CONDITIONS, EMOTIONS,
                                            write_csv, read_csv, contribution_arrays, recorded_branches)
        heads = ('video', 'audio', 'fusion_video', 'fusion_audio')
        base = self.make_rows()
        for n in range(1, 5):
            for selected in itertools.combinations(heads, n):
                rows = []
                for old in base:
                    row = {k: old[k] for k in ('video', 'label', 'prediction')}
                    for emotion in EMOTIONS:
                        value = {'video': old['video_logit_' + emotion],
                                 'audio': old['audio_logit_' + emotion],
                                 'fusion_video': old['interaction_logit_' + emotion],
                                 'fusion_audio': .5 * old['interaction_logit_' + emotion]}
                        for head in selected:
                            row[head + '_logit_' + emotion] = value[head]
                        row['logit_' + emotion] = sum(value[head] for head in selected)
                    row['prediction'] = int(np.argmax([row['logit_' + e] for e in EMOTIONS]))
                    rows.append(row)
                self.assertEqual(recorded_branches(rows), selected)
                logits, terms = contribution_arrays(rows)
                np.testing.assert_allclose(logits, terms.sum(1))
                if selected not in (heads, ('audio',), ('fusion_audio',)):
                    continue
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    for condition in CONDITIONS:
                        write_csv(root / ('test_predictions_' + condition + '.csv'), rows, list(rows[0]))
                    self.assertTrue(analyze_modality_contributions(root, root, 40, []))
                    ablations = read_csv(root / 'modality_contributions' / 'term_ablation_accuracy.csv')
                    if n == 1:
                        self.assertFalse(any(r['configuration'].startswith('drop_') for r in ablations))


if __name__ == '__main__':
    unittest.main()
