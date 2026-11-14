import numpy as np
from .robust_algorithm import fit_train, score_with_fit, training_scale, finite_matrix, SUPPORT_FRACTION
from .group_algorithm import (
    ASSIGNMENT_RULE,
    CRITERION,
    GMM_MAX_ITER,
    GMM_N_INIT,
    GMM_REG_COVAR,
    METHOD,
    MIN_GROUP_FRACTION,
    MIN_GROUP_SIZE,
    assign_windows,
    fit_groups,
    group_calibration,
    validate_group_calibration,
)
from .conformal_algorithm import upper_tail_p
from ..utils.fingerprint import fingerprint

VERSION = 'train-mcd-split-conformal-v1'
VERSION_MULTI = 'train-group-bic-mcd-split-conformal-v1'
MODES = ('single', 'multi')
MULTI_ASSUMPTIONS = (
    "多主群模式以训练窗口的GMM BIC选择群数，再对每群独立做FastMCD；"
    "群数、群中心、协方差与归属规则均在训练后冻结。MCD只适合各群内近对称/椭圆的主群，"
    "群内多峰或训练含未分离污染仍不适用。cal/test只按冻结规则归属并评分，"
    "不参与选群、拟合或阈值。固定训练且各组cal/test独立可交换时，群内上尾split-conformal"
    "提供该群单个新窗口的边际误报控制，不是条件保证、FDR或危险诊断。"
)
MULTI_PROTOCOL = (
    "运行前冻结四特征、mode、alpha与seed。群数只由train的BIC选择；"
    "小群（低于max(20,15%×train)）与奇异群（常量/秩亏/条件数超限）明确拒绝，"
    "不静默合并、丢窗或改用他群校准。cal/test沿用同一归属规则与D²评分，"
    "p只与同群cal比较。归属间隔过小会标注并列，按群编号较小者处理。"
)


def _mode(snapshot):
    config = snapshot.get('config') or {}
    mode = config.get('mode', 'single')
    if mode not in MODES:
        raise ValueError("运行模式仅single或multi")
    return mode


def prepare_snapshot(snapshot):
    mode = _mode(snapshot)
    features = sorted(snapshot['features'], key=lambda f: f['position'])
    codes = [f['code'] for f in features]
    if len(features) != 4 or len(set(codes)) != 4 or len({f['position'] for f in features}) != 4:
        raise ValueError("须固定4个预定义且不重复的特征")
    rows = sorted(snapshot['windows'], key=lambda w: w['code'])
    if not 25 <= len(rows) <= 2000 or len({w['code'] for w in rows}) != len(rows):
        raise ValueError("窗口数25–2000且编号不得重复")
    if len({w['id'] for w in rows}) != len(rows):
        raise ValueError("窗口身份重复，拆分不可交叠")
    groups = {s: [] for s in ['train', 'cal', 'test']}
    for w in rows:
        if w['split'] not in groups:
            raise ValueError("窗口角色仅train/cal/test")
        if set(w['values']) != set(codes):
            raise ValueError("每窗口必须完整包含预定义4量，不得用test选择特征")
        groups[w['split']].append(w)
    if len(groups['train']) < 20 or len(groups['cal']) < 4 or len(groups['test']) < 1:
        raise ValueError("严格拆分要求train≥20/cal≥4/test≥1")
    matrices = {s: finite_matrix([[w['values'][c] for c in codes] for w in g], s) for s, g in groups.items()}
    training_scale(matrices['train'])
    return mode, features, rows, groups, matrices


def _single_analysis(snapshot, features, rows, groups, X, alpha, seed):
    fit = fit_train(X['train'], seed)
    cal = score_with_fit(X['cal'], fit)
    test = score_with_fit(X['test'], fit)
    tail = upper_tail_p(cal, test)
    fit['raw_support_codes'] = [w['code'] for w, m in zip(groups['train'], fit['raw_support_mask']) if m]
    fit['support_codes'] = [w['code'] for w, m in zip(groups['train'], fit['support_mask']) if m]
    scores = [
        {"window": w, "d2": float(d), "p": float(p), "cal_ge_count": int(k), "selected": bool(p <= alpha)}
        for w, d, p, k in zip(groups['test'], test, tail['p'], tail['cal_ge_count'])
    ]
    ecdf_values, ecdf_counts = np.unique(cal, return_counts=True)
    ecdf_y = np.cumsum(ecdf_counts) / len(cal)
    canonical = {"study": snapshot['study'], "features": features, "windows": rows}
    fit_hash = fingerprint({
        "version": VERSION,
        "features": features,
        "train": [{"code": w['code'], "values": w['values']} for w in groups['train']],
        "seed": seed,
        "support_fraction": .8,
    })
    return {
        "algorithm_version": VERSION,
        "mode": "single",
        "study": snapshot['study'],
        "features": features,
        "counts": {s: len(g) for s, g in groups.items()},
        "fit": fit,
        "fit_hash": fit_hash,
        "calibration": [{"window": w, "d2": float(d)} for w, d in zip(groups['cal'], cal)],
        "test": scores,
        "calibration_ecdf": [{"d2": float(d), "cdf": float(y)} for d, y in zip(ecdf_values, ecdf_y)],
        "alpha": alpha,
        "seed": seed,
        "selected_count": sum(x['selected'] for x in scores),
        "resolution": tail['resolution'],
        "tail": tail['tail'],
        "input_snapshot": canonical,
        # Historical runs used a snapshot without config; keep this hash identical for them.
        "input_hash": fingerprint({"version": VERSION, "snapshot": canonical, "alpha": alpha, "seed": seed, "support_fraction": .8}),
        "assumptions": "MCD适用于单峰、近对称/椭圆主群，不适合多峰。定标与MCD只用train；固定训练且cal/test独立可交换时，上尾split-conformal提供边际误报控制，不是条件保证、FDR或危险诊断。",
        "protocol": "四特征与alpha在运行前冻结；不以test选择特征、拟合、阈值或测试分位。筛选仅为人工独立补测线索。",
    }


def _multi_analysis(snapshot, features, rows, groups, X, alpha, seed):
    train_codes = [w['code'] for w in groups['train']]
    fits, selection = fit_groups(X['train'], train_codes, seed)
    cal_entries = assign_windows(X['cal'], fits)
    test_entries = assign_windows(X['test'], fits)
    references = group_calibration(cal_entries, fits)
    validate_group_calibration(references, fits)

    cal_by_group = {group['group_id']: [] for group in fits}
    for entry in cal_entries:
        cal_by_group[entry['group_id']].append(entry['d2'])
    test_by_group = {group['group_id']: [] for group in fits}
    for index, entry in enumerate(test_entries):
        test_by_group[entry['group_id']].append(index)

    test_scores = [None] * len(test_entries)
    for group in fits:
        indices = test_by_group[group['group_id']]
        reference = references[group['group_id']]
        if not indices:
            continue
        tail = upper_tail_p(cal_by_group[group['group_id']], [test_entries[i]['d2'] for i in indices])
        for position, index in enumerate(indices):
            entry = test_entries[index]
            window = groups['test'][index]
            p = float(tail['p'][position])
            test_scores[index] = {
                "window": window,
                "d2": entry['d2'],
                "p": p,
                "cal_ge_count": int(tail['cal_ge_count'][position]),
                "selected": bool(p <= alpha),
                "group_id": entry['group_id'],
                "group_code": entry['group_code'],
                "group_m": reference['m'],
                "group_resolution": reference['resolution'],
                "group_distances": entry['group_distances'],
                "second_group_code": entry['second_group_code'],
                "second_group_d2": entry['second_group_d2'],
                "assignment_margin": entry['assignment_margin'],
                "ambiguous": entry['ambiguous'],
            }

    for group, reference in zip(fits, references):
        group['cal_count'] = reference['m']
        group['test_count'] = len(test_by_group[group['group_id']])
        group['resolution'] = reference['resolution']
        group['calibration_ecdf'] = reference['calibration_ecdf']

    calibration = []
    for window, entry in zip(groups['cal'], cal_entries):
        calibration.append({
            "window": window,
            "d2": entry['d2'],
            "group_id": entry['group_id'],
            "group_code": entry['group_code'],
            "group_distances": entry['group_distances'],
            "second_group_code": entry['second_group_code'],
            "second_group_d2": entry['second_group_d2'],
            "assignment_margin": entry['assignment_margin'],
            "ambiguous": entry['ambiguous'],
        })

    canonical = {"study": snapshot['study'], "features": features, "windows": rows, "config": {"mode": "multi"}}
    group_payload = []
    for group in fits:
        codes = sorted(row['code'] for row in group['train_windows'])
        group_payload.append({
            "group_code": group['group_code'],
            "train_count": group['train_count'],
            "train": [
                {"code": code, "values": next(w['values'] for w in groups['train'] if w['code'] == code)}
                for code in codes
            ],
        })
    fit_hash = fingerprint({
        "version": VERSION_MULTI,
        "mode": "multi",
        "features": features,
        "selection": {
            "method": METHOD,
            "criterion": CRITERION,
            "min_group_size": selection['min_group_size'],
            "min_group_fraction": selection['min_group_fraction'],
            "max_groups": selection['max_groups'],
            "gmm_n_init": GMM_N_INIT,
            "gmm_reg_covar": GMM_REG_COVAR,
            "gmm_max_iter": GMM_MAX_ITER,
            "selected_k": selection['selected_k'],
            "candidates": selection['candidates'],
        },
        "assignment_rule": ASSIGNMENT_RULE,
        "groups": group_payload,
        "seed": seed,
        "support_fraction": SUPPORT_FRACTION,
    })
    for group in fits:
        group["train_fit_hash"] = fingerprint({
            "version": VERSION_MULTI,
            "group_code": group['group_code'],
            "train": next(payload for payload in group_payload if payload['group_code'] == group['group_code'])['train'],
            "seed": seed,
            "support_fraction": SUPPORT_FRACTION,
        })
    input_hash = fingerprint({
        "version": VERSION_MULTI,
        "mode": "multi",
        "snapshot": canonical,
        "alpha": alpha,
        "seed": seed,
        "support_fraction": SUPPORT_FRACTION,
        "selection": {
            "method": METHOD,
            "criterion": CRITERION,
            "min_group_size": MIN_GROUP_SIZE,
            "min_group_fraction": MIN_GROUP_FRACTION,
            "gmm_n_init": GMM_N_INIT,
            "gmm_reg_covar": GMM_REG_COVAR,
            "gmm_max_iter": GMM_MAX_ITER,
        },
        "assignment_rule": ASSIGNMENT_RULE,
    })
    return {
        "algorithm_version": VERSION_MULTI,
        "mode": "multi",
        "study": snapshot['study'],
        "features": features,
        "counts": {s: len(g) for s, g in groups.items()},
        "group_selection": selection,
        "assignment_rule": ASSIGNMENT_RULE,
        "groups": fits,
        "group_count": len(fits),
        "fit_hash": fit_hash,
        "group_fit_hashes": {group['group_code']: group['train_fit_hash'] for group in fits},
        "calibration": calibration,
        "test": test_scores,
        "calibration_ecdf": [],
        "alpha": alpha,
        "seed": seed,
        "selected_count": sum(x['selected'] for x in test_scores),
        "resolution": None,
        "tail": "群内上尾含等号：p=(1+#同群cal D²≥test D²)/(m_group+1)，同值保守",
        "input_snapshot": canonical,
        "input_hash": input_hash,
        "assumptions": MULTI_ASSUMPTIONS,
        "protocol": MULTI_PROTOCOL,
    }


def check_multi_feasibility(snapshot, seed=2026):
    """Train-only group selection plus cal attribution check; no test-driven tuning."""
    mode, features, rows, groups, X = prepare_snapshot({**snapshot, "config": {"mode": "multi"}})
    fits, selection = fit_groups(X['train'], [w['code'] for w in groups['train']], seed)
    references = group_calibration(assign_windows(X['cal'], fits), fits)
    validate_group_calibration(references, fits)
    return {
        "group_count": len(fits),
        "group_sizes": [group['train_count'] for group in fits],
        "cal_sizes": [reference['m'] for reference in references],
        "selected_bic": selection['selected_bic'],
        "candidates": selection['candidates'],
    }


def analyze_deviation(snapshot, alpha=.05, seed=2026):
    if type(alpha) not in (int, float) or not np.isfinite(alpha) or not .01 <= alpha <= .2:
        raise ValueError("预设alpha须在.01–.2之间")
    mode, features, rows, groups, X = prepare_snapshot(snapshot)
    if mode == 'multi':
        return _multi_analysis(snapshot, features, rows, groups, X, alpha, seed)
    return _single_analysis(snapshot, features, rows, groups, X, alpha, seed)
