import numpy as np

def precision_at_k(r, k):
  assert 1 <= k <= r.size
  return (np.asarray(r)[:k]).mean()


def average_precision_at_k(r, k):
    r = np.asarray(r)
    n_rel = r.sum()
    if n_rel == 0:
        return 0.0
    vectorized_precision = np.vectorize(lambda i: precision_at_k(r, i))
    indices = np.arange(1, len(r) + 1)
    precisions = vectorized_precision(
        indices
    )
    score = np.sum(precisions * r)
    return score / min(k, n_rel)


def recall_at_k(r, k):
    r = np.asarray(r)
    n_rel = r.sum()
    if n_rel == 0:
        return 0.0
    return np.sum(r[:k]) / n_rel


def dcg_at_k(r, k):
    r = np.asarray(r)[:k]
    if r.size:
        return np.sum(
            np.subtract(np.power(2, r), 1) / np.log2(np.arange(2, r.size + 2))
        )
    return 0.0


def ndcg_at_k(r, k):
    idcg = dcg_at_k(sorted(r, reverse=True), k)

    if not idcg:
        return 0.0
    return dcg_at_k(r, k) / idcg


def novelty(recommended_items, item_popularity):
    score = 0
    for item in recommended_items:
        popularity = item_popularity.get(item, 0)
        score += np.log2(1/popularity) if popularity > 0 else 0
    return score / len(recommended_items) if any(recommended_items) else 0


def diversity(r, k, item_categories):
    score = 0
    for i in range(len(r)):
        for j in range(i + 1, len(r)):
            item_i = r[i]
            item_j = r[j]
            categories_i = item_categories.get(item_i, set())
            categories_j = item_categories.get(item_j, set())
            if categories_i or categories_j:
                intersection = categories_i.intersection(categories_j)
                union = categories_i.union(categories_j)
                jaccard_distance = 1 - (len(intersection) / len(union))
                score += jaccard_distance

    num_pairs = (k * (k - 1)) / 2
    return score / num_pairs if num_pairs > 0 else 0

def calculate_recommendation_disparity(recommendations, ground_truth, user_groups, k, metric='recall', delta=0.05):
    """
    Calcula la "Disparidad de Cobertura de la Recomendación" entre grupos.

    Args:
        recommendations (dict): {user_id: [item_rec_1, item_rec_2, ...]}
        ground_truth (dict): {user_id: [item_rel_1, item_rel_2, ...]}
        user_groups (dict): {user_id: 'group_label'}
        k (int): El corte para @K.
        metric (str): 'recall' (Cobertura) o 'precision' (Tasa de Aceptación).
        delta (float): Umbral de disparidad absoluta para ser considerado "sesgado".

    Returns:
        dict: Un reporte con las métricas por grupo y las disparidades.
    """
    
    # 1. Calcular métricas por usuario y agruparlas
    group_scores = {} 

    for user_id, recs in recommendations.items():
        if user_id not in ground_truth or user_id not in user_groups:
            continue
            
        group = user_groups[user_id]
        recs_k = recs[:k]
        truth_set = set(ground_truth[user_id])
        
        if metric == 'recall' and not truth_set:
            continue
            
        score = 0.0
        if metric == 'recall':
            score = recall_at_k(recs_k, truth_set)
        elif metric == 'precision':
            score = precision_at_k(recs_k, truth_set)
        else:
            raise ValueError("La métrica debe ser 'recall' o 'precision'")
            
        if group not in group_scores:
            group_scores[group] = []
        group_scores[group].append(score)

    # 2. Calcular métricas promedio por grupo
    avg_group_metrics = {}
    for group, scores in group_scores.items():
        if scores:
            avg_group_metrics[group] = np.mean(scores)
        else:
            avg_group_metrics[group] = 0.0

    # 3. Calcular disparidades entre pares de grupos
    report = {
        "metric_name": f"{metric}@k (k={k})",
        "avg_group_metrics": avg_group_metrics,
        "disparities": [],
        "is_biased_by_delta": False,
        "delta_threshold": delta
    }
    
    group_names = list(avg_group_metrics.keys())
    
    for i in range(len(group_names)):
        for j in range(i + 1, len(group_names)):
            group_a = group_names[i]
            group_b = group_names[j]
            
            metric_a = avg_group_metrics[group_a]
            metric_b = avg_group_metrics[group_b]
            
            abs_disparity = abs(metric_a - metric_b)
            
            min_metric = min(metric_a, metric_b)
            max_metric = max(metric_a, metric_b)
            relative_ratio = min_metric / max_metric if max_metric > 0 else 1.0 
            
            disparity_info = {
                "pair": (group_a, group_b),
                "absolute_difference": abs_disparity,
                "relative_ratio": relative_ratio,
                "disadvantaged_group": group_a if metric_a < metric_b else group_b,
                "violates_delta": abs_disparity > delta
            }
            
            report["disparities"].append(disparity_info)
            
            if abs_disparity > delta:
                report["is_biased_by_delta"] = True

    return report