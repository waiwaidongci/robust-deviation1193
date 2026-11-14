"""Train-only multi-group discovery, frozen per-group MCD fits and assignment rule.

Group count is selected only from training windows with a seeded full-covariance
Gaussian mixture BIC search. Calibration and test rows are never used to choose
groups, fit centres/covariances or tune the assignment rule.
"""
import numpy as np
from sklearn.mixture import GaussianMixture
from .robust_algorithm import fit_train, score_with_fit, training_scale

MAX_GROUPS = 5
MIN_GROUP_SIZE = 20
MIN_GROUP_FRACTION = .15
MIN_CAL_GROUP = 4
GMM_N_INIT = 5
GMM_MAX_ITER = 300
GMM_REG_COVAR = 1e-6
METHOD = "GaussianMixture(covariance_type=full)+BIC"
CRITERION = "BIC最小；并列取较小群数；候选与选择只使用train窗口"
ASSIGNMENT_RULE = {
    "method": "minimum_reweighted_mahalanobis_squared",
    "tie_break": "群编号较小者",
    "basis": "对全部冻结群模型计算重加权D²，归入D²最小的群；最近与次近群D²之差为归属间隔。",
    "frozen": "归属规则在训练拟合后冻结；cal/test只按该规则归属和评分，不参与选群、中心、协方差或阈值。",
    "ambiguity": "归属间隔≤1e-9记为并列；并列按群编号较小者处理，并在结果中标注。",
}
MIN_SIZE_RULE = "小群和奇异群明确拒绝；不静默合并、丢弃或借用他群校准"


def _select_group_count(z, seed):
    if type(seed) is not int or not 0 <= seed <= 2147483647:
        raise ValueError("seed须为非负32位整数")
    n = z.shape[0]
    largest = max(1, min(MAX_GROUPS, n // MIN_GROUP_SIZE))
    minimum = max(MIN_GROUP_SIZE, int(np.ceil(MIN_GROUP_FRACTION * n)))
    candidates, labels_by_k = [], {}
    for k in range(1, largest + 1):
        try:
            mixture = GaussianMixture(
                n_components=k,
                covariance_type="full",
                random_state=seed,
                n_init=GMM_N_INIT,
                max_iter=GMM_MAX_ITER,
                reg_covar=GMM_REG_COVAR,
            ).fit(z)
            labels = mixture.predict(z)
            sizes = sorted((int(x) for x in np.bincount(labels, minlength=k)), reverse=True)
            bic = float(mixture.bic(z))
            if not np.isfinite(bic):
                raise ValueError("BIC不是有限数值")
            eligible = all(size >= minimum for size in sizes)
            reason = "" if eligible else f"最小群{sizes[-1]}/{n}条低于阈值{minimum}条（max({MIN_GROUP_SIZE}, {MIN_GROUP_FRACTION:.0%}×train))"
        except Exception as exc:
            labels, sizes, bic, eligible = None, [], None, False
            reason = f"候选拟合失败：{exc}"
        candidates.append({"k": k, "bic": bic, "sizes": sizes, "eligible": eligible, "reason": reason})
        if labels is not None:
            labels_by_k[k] = labels
    usable = [candidate for candidate in candidates if candidate["bic"] is not None]
    if not usable:
        raise ValueError("训练窗口无法完成任何分组候选拟合，拒绝多主群研究")
    best = min(usable, key=lambda candidate: (candidate["bic"], candidate["k"]))
    for candidate in candidates:
        candidate["delta_bic"] = float(candidate["bic"] - best["bic"]) if candidate["bic"] is not None else None
        candidate["selected"] = candidate["k"] == best["k"]
    if not best["eligible"]:
        detail = " / ".join(f"G{i + 1}={size}" for i, size in enumerate(best["sizes"]))
        raise ValueError(
            f"训练BIC选择{best['k']}群，但出现小群（{detail}；阈值{minimum}条）：明确拒绝，不静默合并或丢窗"
        )
    selection = {
        "method": METHOD,
        "criterion": CRITERION,
        "train_only": True,
        "max_groups": largest,
        "min_group_size": minimum,
        "min_group_fraction": MIN_GROUP_FRACTION,
        "gmm_n_init": GMM_N_INIT,
        "gmm_reg_covar": GMM_REG_COVAR,
        "candidates": candidates,
        "selected_k": best["k"],
        "selected_bic": best["bic"],
        "note": "群数只由train窗口决定；cal/test不参与选群或拟合。" + MIN_SIZE_RULE + "。",
    }
    return selection, labels_by_k[best["k"]]


def fit_groups(train, train_codes, seed=2026):
    """Freeze per-group MCD models using train only; return groups and BIC selection."""
    z, global_mean, global_scale, global_rank = training_scale(train)
    selection, labels = _select_group_count(z, seed)
    clusters = []
    for label in range(selection["selected_k"]):
        indices = np.flatnonzero(labels == label)
        centroid = tuple(float(v) for v in z[indices].mean(axis=0))
        clusters.append((len(indices), centroid, indices))
    # Canonical group order is independent of GMM label permutation and row order.
    clusters.sort(key=lambda item: (-item[0], item[1]))
    groups = []
    for position, (count, _centroid, indices) in enumerate(clusters):
        group_code = f"G{position + 1}"
        try:
            fit = fit_train(train[indices], seed)
        except ValueError as exc:
            raise ValueError(f"群{group_code}拟合失败（{count}条训练窗口）：{exc}；明确拒绝奇异群，不静默丢窗") from exc
        codes = [train_codes[index] for index in indices]
        fit.update({
            "group_id": position,
            "group_code": group_code,
            "train_count": int(count),
            "train_windows": [
                {"code": codes[i], "raw_support": bool(fit["raw_support_mask"][i]), "support": bool(fit["support_mask"][i]), "d2": float(fit["train_distances"][i])}
                for i in range(count)
            ],
            "train_fit_hash": None,
            "cal_count": 0,
            "test_count": 0,
            "resolution": None,
            "calibration_ecdf": [],
        })
        groups.append(fit)
    selection["group_sizes"] = [group["train_count"] for group in groups]
    selection["global_training_rank"] = int(global_rank)
    selection["global_scaler_mean"] = [float(v) for v in global_mean]
    selection["global_scaler_scale"] = [float(v) for v in global_scale]
    return groups, selection


def assign_windows(values, groups):
    """Apply the frozen assignment rule: nearest final reweighted D², lowest group id on ties."""
    distances = np.vstack([score_with_fit(values, group) for group in groups])
    entries = []
    for column in range(distances.shape[1]):
        order = np.argsort(distances[:, column], kind="stable")
        group_id = int(order[0])
        second_id = int(order[1]) if len(order) > 1 else None
        d2 = float(distances[group_id, column])
        second_d2 = float(distances[second_id, column]) if second_id is not None else None
        margin = float(second_d2 - d2) if second_d2 is not None else None
        entries.append({
            "d2": d2,
            "group_id": group_id,
            "group_code": groups[group_id]["group_code"],
            "group_distances": [
                {"group_id": gid, "group_code": group["group_code"], "d2": float(distances[gid, column])}
                for gid, group in enumerate(groups)
            ],
            "second_group_id": second_id,
            "second_group_code": groups[second_id]["group_code"] if second_id is not None else None,
            "second_group_d2": second_d2,
            "assignment_margin": margin,
            "ambiguous": bool(margin is not None and margin <= 1e-9),
        })
    return entries


def group_calibration(calibration, groups):
    """Build per-group frozen calibration references from assigned cal scores."""
    by_group = {group["group_id"]: [] for group in groups}
    for entry in calibration:
        by_group[entry["group_id"]].append(entry["d2"])
    references = []
    for group in groups:
        values = sorted(by_group[group["group_id"]])
        unique, counts = np.unique(values, return_counts=True)
        cdf = np.cumsum(counts) / len(values) if values else np.array([])
        references.append({
            "group_id": group["group_id"],
            "group_code": group["group_code"],
            "m": len(values),
            "resolution": 1 / (len(values) + 1) if values else None,
            "calibration_ecdf": [{"d2": float(d2), "cdf": float(y)} for d2, y in zip(unique, cdf)],
        })
    return references


def validate_group_calibration(references, groups):
    for reference in references:
        if reference["m"] < MIN_CAL_GROUP:
            raise ValueError(
                f"群{reference['group_code']}仅分到{reference['m']}条校准窗口（至少{MIN_CAL_GROUP}条）："
                "明确拒绝，不用他群校准或测试分数替代"
            )
