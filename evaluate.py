import json

import numpy as np
import pandas as pd
import tqdm

from metrics import (
    average_precision_at_k,
    diversity,
    ndcg_at_k,
    novelty,
    precision_at_k,
    recall_at_k,
)


def evaluate_model(
    n,
    user_groups,
    user_items_test: dict,
    item_popularity: dict,
    item_categories: dict,
    get_recommendations,
    alpha=0.02,
):
    """
    Evalúa el modelo calculando métricas globales y de fairness.

    Args:
        n (int): El 'k' para las métricas @k.
        user_groups (dict): {user_id: 'group_label'}
        delta (float): Umbral de disparidad para considerar "sesgado".
    """

    groups = list(set(user_groups.values()))
    group_names_to_consider = [
        "Male",
        "Female",
        "<18",
        "18-24",
        "25-34",
        "35-44",
        "45-54",
        "55-64",
        "65+",
    ]

    group_scores = {
        "recall": {},
        "precision": {},
        "map": {},
        "ndcg": {},
        "novelty": {},
        "diversity": {},
        "category_counts": {},
    }

    all_recalls = []
    all_maps = []
    all_ndcgs = []
    all_novelties = []
    all_diversities = []
    all_precisions = []

    for user_id in tqdm.tqdm(user_items_test.keys(), desc="Evaluando usuarios"):

        # 1. Obtener datos del usuario
        if user_id not in user_groups:
            continue

        group = user_groups[user_id]
        if group not in group_names_to_consider:
            continue

        truth_items = user_items_test.get(user_id, set())

        # 2. Obtener recomendaciones y vector de relevancia
        rec = get_recommendations(user_id, n)

        rel_vector = np.isin(rec, truth_items, assume_unique=True).astype(int)

        # 3. Calcular métricas individuales
        user_recall = recall_at_k(rel_vector, n)
        user_precision = precision_at_k(rel_vector, n)
        user_map = average_precision_at_k(rel_vector, n)
        user_ndcg = ndcg_at_k(rel_vector, n)
        user_novelty = novelty(rec, item_popularity)
        user_diversity = diversity(rec, n, item_categories)

        # 4. Almacenar para métricas globales
        all_recalls.append(user_recall)
        all_precisions.append(user_precision)
        all_maps.append(user_map)
        all_ndcgs.append(user_ndcg)
        all_novelties.append(user_novelty)
        all_diversities.append(user_diversity)

        # 5. Almacenar métricas de fairness por grupo
        if group not in group_scores["recall"]:
            group_scores["recall"][group] = []
            group_scores["precision"][group] = []
            group_scores["map"][group] = []
            group_scores["ndcg"][group] = []
            group_scores["novelty"][group] = []
            group_scores["diversity"][group] = []
            group_scores["category_counts"][group] = {}

        group_scores["recall"][group].append(user_recall)
        group_scores["precision"][group].append(user_precision)
        group_scores["map"][group].append(user_map)
        group_scores["ndcg"][group].append(user_ndcg)
        group_scores["novelty"][group].append(user_novelty)
        group_scores["diversity"][group].append(user_diversity)

        for item_id in rec:
            categories: set = item_categories.get(item_id, ("UNKNOWN",))
            for category in categories:
                group_scores["category_counts"][group][category] = (
                    group_scores["category_counts"][group].get(category, 0) + 1
                )

    # --- 6. Calcular Métricas Promedio (Globales) ---

    metrics_global = {
        "mean_recall": np.mean(all_recalls) if all_recalls else 0.0,
        "mean_precision": np.mean(all_precisions) if all_precisions else 0.0,
        "mean_ap (MAP)": np.mean(all_maps) if all_maps else 0.0,
        "mean_ndcg": np.mean(all_ndcgs) if all_ndcgs else 0.0,
        "mean_novelty": np.mean(all_novelties) if all_novelties else 0.0,
        "mean_diversity": np.mean(all_diversities) if all_diversities else 0.0,
        "num_users_evaluated": len(all_recalls),
    }

    # --- 7. Calcular Métricas de Fairness (Disparidad) ---

    fairness_report = {
        "delta_threshold": alpha,
        "group_averages": {},
        "disparity_reports": [],
        "category_disparity_report": {},
    }

    avg_recall_group = {g: np.mean(s) for g, s in group_scores["recall"].items() if s}
    avg_precision_group = {
        g: np.mean(s) for g, s in group_scores["precision"].items() if s
    }
    avg_map_group = {g: np.mean(s) for g, s in group_scores["map"].items() if s}
    avg_ndcg_group = {g: np.mean(s) for g, s in group_scores["ndcg"].items() if s}
    avg_novelty_group = {g: np.mean(s) for g, s in group_scores["novelty"].items() if s}
    avg_diversity_group = {
        g: np.mean(s) for g, s in group_scores["diversity"].items() if s
    }

    group_category_proportions = {}
    for group, counts in group_scores["category_counts"].items():
        total_recs = sum(counts.values())
        if total_recs > 0:
            group_category_proportions[group] = {
                cat: count / total_recs for cat, count in counts.items()
            }

    fairness_report["group_averages"] = {
        g: {
            "recall": avg_recall_group.get(g, 0.0),
            "precision": avg_precision_group.get(g, 0.0),
            "MAP": avg_map_group.get(g, 0.0),
            "nDCG": avg_ndcg_group.get(g, 0.0),
            "novelty": avg_novelty_group.get(g, 0.0),
            "diversity": avg_diversity_group.get(g, 0.0),
            "count": len(group_scores["recall"].get(g, [])),
        }
        for g in avg_recall_group.keys()
    }

    # Comparar pares de grupos
    group_names_to_consider = [
        "Male",
        "Female",
        "<18",
        "18-24",
        "25-34",
        "35-44",
        "45-54",
        "55-64",
        "65+",
    ]
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            g_a = groups[i]
            g_b = groups[j]

            if (g_a not in group_names_to_consider) or (
                g_b not in group_names_to_consider
            ):
                continue

            disp_recall = calc_disparity(g_a, g_b, avg_recall_group)
            disp_precision = calc_disparity(g_a, g_b, avg_precision_group)
            disp_map = calc_disparity(g_a, g_b, avg_map_group)
            disp_ndcg = calc_disparity(g_a, g_b, avg_ndcg_group)
            disp_novelty = calc_disparity(g_a, g_b, avg_novelty_group)
            disp_diversity = calc_disparity(g_a, g_b, avg_diversity_group)

            pair_report = {
                "pair": (g_a, g_b),
                "recall_disparity": {
                    "value": disp_recall,
                    "bias": f"{is_biased(disp_recall, alpha)}",
                    "toward": (
                        g_a
                        if avg_recall_group.get(g_a, 0.0)
                        > avg_recall_group.get(g_b, 0.0)
                        else g_b
                    ),
                },
                "precision_disparity": {
                    "value": disp_precision,
                    "bias": f"{is_biased(disp_precision, alpha)}",
                    "toward": (
                        g_a
                        if avg_precision_group.get(g_a, 0.0)
                        > avg_precision_group.get(g_b, 0.0)
                        else g_b
                    ),
                },
                "map_disparity": {
                    "value": disp_map,
                    "bias": f"{is_biased(disp_map, alpha)}",
                    "toward": (
                        g_a
                        if avg_map_group.get(g_a, 0.0) > avg_map_group.get(g_b, 0.0)
                        else g_b
                    ),
                },
                "ndcg_disparity": {
                    "value": disp_ndcg,
                    "bias": f"{is_biased(disp_ndcg, alpha)}",
                    "toward": (
                        g_a
                        if avg_ndcg_group.get(g_a, 0.0) > avg_ndcg_group.get(g_b, 0.0)
                        else g_b
                    ),
                },
                "novelty_disparity": {
                    "value": disp_novelty,
                    "bias": f"{is_biased(disp_novelty, alpha)}",
                    "toward": (
                        g_a
                        if avg_novelty_group.get(g_a, 0.0)
                        > avg_novelty_group.get(g_b, 0.0)
                        else g_b
                    ),
                },
                "diversity_disparity": {
                    "value": disp_diversity,
                    "bias": f"{is_biased(disp_diversity, alpha)}",
                    "toward": (
                        g_a
                        if avg_diversity_group.get(g_a, 0.0)
                        > avg_diversity_group.get(g_b, 0.0)
                        else g_b
                    ),
                },
            }
            fairness_report["disparity_reports"].append(pair_report)

            category_disparities = {}
            # Obtener todas las categorías vistas por ambos grupos
            all_categories = set(group_category_proportions.get(g_a, {}).keys()) | set(
                group_category_proportions.get(g_b, {}).keys()
            )

            for cat in all_categories:
                prop_a = group_category_proportions.get(g_a, {}).get(cat, 0.0)
                prop_b = group_category_proportions.get(g_b, {}).get(cat, 0.0)
                # La disparidad es la diferencia en la proporción de recomendaciones de esa categoría
                category_disparities[cat] = (
                    min(prop_a, prop_b) / max(prop_a, prop_b)
                    if max(prop_a, prop_b) > 0
                    else 1.0
                )

            fairness_report["category_disparity_report"][f"{g_a}_vs_{g_b}"] = {}
            for cat in category_disparities:
                fairness_report["category_disparity_report"][f"{g_a}_vs_{g_b}"][cat] = {
                    "value": category_disparities[cat],
                    "bias": f"{is_biased(category_disparities[cat], alpha)}",
                    "toward": (
                        g_a
                        if group_category_proportions.get(g_a, {}).get(cat, 0.0)
                        > group_category_proportions.get(g_b, {}).get(cat, 0.0)
                        else g_b
                    ),
                }

    return metrics_global, fairness_report


def calc_disparity(a, b, metric):
    r_a, r_b = metric[a], metric[b]
    return min(r_a, r_b) / max(r_a, r_b)


def is_biased(disparity, tolerance):
    return disparity < 1 - tolerance


def get_metrics(
    df_train,
    df_test,
    get_recommendations,
    k=10,
    alpha=0.02,
    use_age_group=False,
):
    
    user_items = {}
    for row in df_train.itertuples():
        if row[1] not in user_items:
            user_items[row[1]] = []
        user_items[row[1]].append(row[2])
    user_items_test = {}
    for row in df_test.itertuples():
        if row[1] not in user_items_test:
            user_items_test[row[1]] = []
        user_items_test[row[1]].append(row[2])

    # Drop all users that are not in the training set
    user2row = {user_id: matrix_row for matrix_row, user_id in enumerate(user_items.keys())}
    user_items_test = {user: items for user, items in user_items_test.items() if user in user2row}

    item_interaction_counts = df_train['itemid'].value_counts()
    user_count = df_train['userid'].nunique()
    item_popularity = (item_interaction_counts / user_count).to_dict()
    
    animes = pd.read_csv("clean_data/animes.csv")
    metadata = animes[['uid', 'genre']]
    item_categories: dict[int, set[str | None]] = {}
    for row in metadata.itertuples():
        if isinstance(row.genre, str):
            genre_list = set(g.strip().replace("'", "") for g in row.genre[1:-1].split(','))
        else:
            genre_list: set = set()

        if isinstance(row.uid, int):
            item_categories[row.uid] = genre_list
        else:
            raise ValueError("Unexpected non-integer uid")
    
    profiles = pd.read_csv("clean_data/profiles.csv")

    if use_age_group:
        profiles["age_group"] = profiles["birthday"].apply(age_to_group)

    user_groups_map = {}
    # user_groups_map id a sexo
    for row in profiles.itertuples():
        user_id = row.user_id
        user_groups_map[user_id] = row.gender if not use_age_group else row.age_group

    global_metrics, fairness_results = evaluate_model(
        k,
        user_groups_map,
        user_items_test,
        item_popularity,
        item_categories,
        get_recommendations,
        alpha=alpha,
    )

    print("--- Métricas Globales de Evaluación ---")
    print(json.dumps(global_metrics, indent=2))

    print("\n--- Reporte de Fairness (Disparidad de Grupo) ---")
    print(json.dumps(fairness_results, indent=2))

    print(f"\nRecall Global: {global_metrics['mean_recall']:.4f}")
    print(f"MAP Global: {global_metrics['mean_ap (MAP)']:.4f}")
    # print(f"¿Es sesgado (Recall)?: {fairness_results['is_biased_recall']}")
    # print(f"¿Es sesgado (Precision)?: {fairness_results['is_biased_precision']}")


def age_to_group(birthday_str):
    """It's either '', 'YYYY' or 'DD-MM-YYYY'"""

    if not isinstance(birthday_str, str) or birthday_str.strip() == "":
        return "Unknown"
    try:
        year = int(birthday_str[-4:])
        age = 2025 - year
        if age < 18:
            return "<18"
        elif 18 <= age < 25:
            return "18-24"
        elif 25 <= age < 35:
            return "25-34"
        elif 35 <= age < 45:
            return "35-44"
        elif 45 <= age < 55:
            return "45-54"
        elif 55 <= age < 65:
            return "55-64"
        else:
            return "65+"
    except ValueError:
        return "Unknown"
