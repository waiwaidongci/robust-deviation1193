import unittest
import numpy as np
from backend.src.services.analysis_service import analyze_deviation, check_multi_feasibility

# Run from repository root: python3 -m unittest discover -s backend/tests

MEAN = np.array([19.5, 47., 18., .35])
SCALE = np.array([1.5, 5., 4., .12])
R = np.array([[1, .55, .25, .2], [.55, 1, .35, .1], [.25, .35, 1, .3], [.2, .1, .3, 1]])
CODES = ['temperature', 'humidity', 'particles', 'variation']


def mixed_windows(seed=300, main_n=250, second_n=250):
    rng = np.random.default_rng(seed)
    main = MEAN + rng.multivariate_normal(np.zeros(4), R, main_n) * SCALE
    second = MEAN + np.array([5, -13, 10, .25]) + rng.multivariate_normal(np.zeros(4), R, second_n) * SCALE
    X = np.vstack([main, second])
    X = X[rng.permutation(len(X))]
    return windows(X, 'MX')


def windows(X, prefix):
    out = []
    for i, row in enumerate(X):
        split = 'train' if i < 200 else 'cal' if i < 350 else 'test'
        out.append({'id': i + 1, 'code': f'{prefix}-{i + 1:03}', 'split': split, 'values': {c: float(v) for c, v in zip(CODES, row)}})
    return out


def snapshot(rows, mode='multi'):
    return {'study': {'id': 1, 'code': 'SL-X'}, 'config': {'mode': mode}, 'features': [{'id': i + 1, 'code': c, 'position': i} for i, c in enumerate(CODES)], 'windows': rows}


def scores(result):
    return [(x['window']['code'], round(x['p'], 12), x['cal_ge_count'], x['group_code']) for x in result['test']]


class MultiGroupTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = mixed_windows()
        cls.result = analyze_deviation(snapshot(cls.rows), .05, 2026)

    def test_two_groups_selected_from_train_only(self):
        result = self.result
        self.assertEqual(result['mode'], 'multi')
        self.assertEqual(result['group_count'], 2)
        self.assertEqual(sorted(result['group_selection']['group_sizes']), [100, 100])
        self.assertEqual(len(result['groups']), 2)
        for group in result['groups']:
            self.assertGreaterEqual(group['cal_count'], 4)
            self.assertGreater(group['condition_number'], 0)
            self.assertLessEqual(group['condition_number'], 1e10)
            self.assertIsNotNone(group['train_fit_hash'])

    def test_test_uses_own_group_calibration(self):
        result = self.result
        for item in result['test']:
            self.assertEqual(item['group_m'], next(g['cal_count'] for g in result['groups'] if g['group_code'] == item['group_code']))
            self.assertGreaterEqual(item['p'], 1 / (item['group_m'] + 1))
            self.assertEqual(item['selected'], item['p'] <= .05)
            self.assertEqual(item['group_code'], min(item['group_distances'], key=lambda x: (x['d2'], x['group_id']))['group_code'])
            self.assertAlmostEqual(item['assignment_margin'], item['second_group_d2'] - item['d2'])

    def test_row_order_and_feature_order_do_not_change_result(self):
        shuffled = list(reversed(self.rows))
        other = analyze_deviation(snapshot(shuffled), .05, 2026)
        self.assertEqual(self.result['fit_hash'], other['fit_hash'])
        self.assertEqual(self.result['input_hash'], other['input_hash'])
        self.assertEqual(scores(self.result), scores(other))
        self.assertEqual([(g['group_code'], g['train_windows'][0]['code']) for g in self.result['groups']], [(g['group_code'], g['train_windows'][0]['code']) for g in other['groups']])

    def test_changed_test_values_do_not_change_fit_or_calibration(self):
        changed = [dict(row) for row in self.rows]
        for row in changed:
            if row['split'] == 'test':
                row['values'] = {code: value + delta for (code, value), delta in zip(row['values'].items(), [8, -8, 6, .3])}
        other = analyze_deviation(snapshot(changed), .05, 2026)
        self.assertEqual(self.result['fit_hash'], other['fit_hash'])
        self.assertEqual(self.result['group_selection']['selected_k'], other['group_selection']['selected_k'])
        self.assertEqual([(x['window']['code'], x['d2']) for x in self.result['calibration']], [(x['window']['code'], x['d2']) for x in other['calibration']])
        self.assertNotEqual(self.result['input_hash'], other['input_hash'])

    def test_changed_alpha_does_not_change_scores(self):
        other = analyze_deviation(snapshot(self.rows), .1, 2026)
        self.assertEqual([(a['window']['code'], a['d2'], a['p']) for a in self.result['test']], [(b['window']['code'], b['d2'], b['p']) for b in other['test']])
        self.assertNotEqual(self.result['input_hash'], other['input_hash'])

    def test_single_mode_still_available_and_uses_one_fit(self):
        single = analyze_deviation(snapshot(self.rows, 'single'), .05, 2026)
        self.assertEqual(single['mode'], 'single')
        self.assertIn('fit', single)
        self.assertEqual(single['fit']['raw_support_count'] + single['fit']['support_count'] > 0, True)

    def test_historical_snapshot_without_config_is_single(self):
        old = snapshot(self.rows, 'single')
        old.pop('config')
        result = analyze_deviation(old, .05, 2026)
        self.assertEqual(result['mode'], 'single')


class RejectionTest(unittest.TestCase):
    def test_small_group_is_rejected_not_silently_dropped(self):
        rows = mixed_windows(seed=301)
        X = np.vstack([
            np.array([[row['values'][c] for c in CODES] for row in rows]),
        ])
        # Replace 18 training rows with a tiny far-away group; BIC then yields a small group.
        X[:18] = MEAN + np.array([35, 70, 55, 2.5])
        rows = windows(X, 'SM')
        with self.assertRaisesRegex(ValueError, '小群'):
            analyze_deviation(snapshot(rows), .05, 2026)

    def test_degenerate_training_is_rejected(self):
        X = np.array([[row['values'][c] for c in CODES] for row in mixed_windows(seed=302)])
        X[:200, 3] = .1 * X[:200, 0] + .01 * X[:200, 1]
        with self.assertRaisesRegex(ValueError, '秩亏'):
            analyze_deviation(snapshot(windows(X, 'DG')), .05, 2026)

    def test_group_with_too_little_calibration_is_rejected(self):
        rng = np.random.default_rng(303)
        train_a = MEAN + rng.multivariate_normal(np.zeros(4), R, 100) * SCALE
        train_b = MEAN + np.array([6, -14, 11, .3]) + rng.multivariate_normal(np.zeros(4), R, 100) * SCALE
        cal_a = MEAN + rng.multivariate_normal(np.zeros(4), R, 146) * SCALE
        cal_b = MEAN + np.array([6, -14, 11, .3]) + rng.multivariate_normal(np.zeros(4), R, 3) * SCALE
        test = rng.normal(size=(100, 4)) * .1
        X = np.vstack([train_a, train_b, cal_a, cal_b, test])
        rows = windows(X, 'CA')
        with self.assertRaisesRegex(ValueError, '校准窗口'):
            analyze_deviation(snapshot(rows), .05, 2026)

    def test_feasibility_never_reads_test_windows_for_selection(self):
        rows = mixed_windows(seed=304)
        feasibility = check_multi_feasibility(snapshot(rows))
        self.assertEqual(feasibility['group_count'], 2)
        self.assertEqual(sum(feasibility['group_sizes']), 200)
        test_windows = [row for row in rows if row['split'] == 'test']
        mutated = snapshot([dict(row) for row in rows])
        for row in mutated['windows']:
            if row['code'] in {window['code'] for window in test_windows}:
                row['values'] = {code: value + 50 for code, value in row['values'].items()}
        self.assertEqual(feasibility, check_multi_feasibility(mutated))


if __name__ == '__main__':
    unittest.main()
