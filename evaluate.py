import numpy as np
import pandas as pd
import json
from metrics import precision_at_k, ndcg_at_k, novelty, diversity, recall_at_k, average_precision_at_k

def evaluate_model(n, user_groups, user_items_test: dict, item_popularity: dict, item_categories: dict, get_recommendations, delta=0.05):
    """
    Evalúa el modelo calculando métricas globales y de fairness.
    
    Args:
        n (int): El 'k' para las métricas @k.
        user_groups (dict): {user_id: 'group_label'}
        delta (float): Umbral de disparidad para considerar "sesgado".
    """
    
    group_scores = {
        'recall': {},  
        'precision': {} 
    }
    
    all_recalls = []
    all_aps = []
    all_ndcgs = []
    all_novelties = []
    all_diversities = []
    all_precisions = []
    
    for user_id in user_items_test.keys():
        
        # 1. Obtener datos del usuario
        if user_id not in user_groups:
            continue
            
        group = user_groups[user_id]
        truth_items = user_items_test.get(user_id, set())
        
        # 2. Obtener recomendaciones y vector de relevancia
        rec = get_recommendations(user_id, n)
        
        rel_vector = np.isin(rec, truth_items, assume_unique=True).astype(int)

        # 3. Calcular métricas individuales
        user_recall = recall_at_k(rel_vector, n)
        user_precision = precision_at_k(rel_vector, n)
        user_ap = average_precision_at_k(rel_vector, n)
        user_ndcg = ndcg_at_k(rel_vector, n)
        user_novelty = novelty(rec, item_popularity)
        user_diversity = diversity(rec, n, item_categories)

        # 4. Almacenar para métricas globales
        all_recalls.append(user_recall)
        all_precisions.append(user_precision)
        all_aps.append(user_ap)
        all_ndcgs.append(user_ndcg)
        all_novelties.append(user_novelty)
        all_diversities.append(user_diversity)

        # 5. Almacenar métricas de fairness por grupo
        if group not in group_scores['recall']:
            group_scores['recall'][group] = []
            group_scores['precision'][group] = []
            
        group_scores['recall'][group].append(user_recall)
        group_scores['precision'][group].append(user_precision)

    # --- 6. Calcular Métricas Promedio (Globales) ---
    
    metrics_global = {
        'mean_recall': np.mean(all_recalls) if all_recalls else 0.0,
        'mean_precision': np.mean(all_precisions) if all_precisions else 0.0,
        'mean_ap (MAP)': np.mean(all_aps) if all_aps else 0.0,
        'mean_ndcg': np.mean(all_ndcgs) if all_ndcgs else 0.0,
        'mean_novelty': np.mean(all_novelties) if all_novelties else 0.0,
        'mean_diversity': np.mean(all_diversities) if all_diversities else 0.0,
        'num_users_evaluated': len(all_recalls)
    }
    
    # --- 7. Calcular Métricas de Fairness (Disparidad) ---
    
    fairness_report = {
        'delta_threshold': delta,
        'is_biased_recall': 0,
        'is_biased_precision': 0,
        'group_averages': {},
        'disparity_reports': []
    }

    avg_recall_group = {g: np.mean(s) for g, s in group_scores['recall'].items() if s}
    avg_precision_group = {g: np.mean(s) for g, s in group_scores['precision'].items() if s}
    
    fairness_report['group_averages'] = {
        g: {
            'recall (Cobertura)': avg_recall_group.get(g, 0.0),
            'precision (Tasa Aceptación)': avg_precision_group.get(g, 0.0),
            'count': len(group_scores['recall'].get(g, []))
        } for g in avg_recall_group.keys()
    }

    # Comparar pares de grupos
    group_names = list(avg_recall_group.keys())
    for i in range(len(group_names)):
        for j in range(i + 1, len(group_names)):
            g_a = group_names[i]
            g_b = group_names[j]
            
            r_a, r_b = avg_recall_group[g_a], avg_recall_group[g_b]
            abs_disp_recall = abs(r_a - r_b)
            
            p_a, p_b = avg_precision_group[g_a], avg_precision_group[g_b]
            abs_disp_precision = abs(p_a - p_b)

            pair_report = {
                'pair': (g_a, g_b),
                'recall_disparity': abs_disp_recall,
                'precision_disparity': abs_disp_precision,
                'biased_recall (Cobertura)': 1 if abs_disp_recall > delta else 0,
                'biased_precision (Tasa Aceptación)': 1 if abs_disp_precision > delta else 0
            }
            fairness_report['disparity_reports'].append(pair_report)
            
            if abs_disp_recall > delta:
                fairness_report['is_biased_recall'] = 1
            if abs_disp_precision > delta:
                fairness_report['is_biased_precision'] = 1
    
    return metrics_global, fairness_report

def get_metrics(user_items_test, item_popularity, item_categories, get_recommendations, k=10, delta=0.05):
    profiles = pd.read_csv('clean_data/profiles.csv')

    user_groups_map = {}
    # user_groups_map id a sexo
    for row in profiles.itertuples():
        user_id = row.user_id
        user_groups_map[user_id] = row.gender

    global_metrics, fairness_results = evaluate_model(k, user_groups_map, user_items_test, item_popularity, item_categories, get_recommendations, delta=0.05)

    print("--- Métricas Globales de Evaluación ---")
    print(json.dumps(global_metrics, indent=2))

    print("\n--- Reporte de Fairness (Disparidad de Grupo) ---")
    print(json.dumps(fairness_results, indent=2))

    print(f"\nRecall Global: {global_metrics['mean_recall']:.4f}")
    print(f"MAP Global: {global_metrics['mean_ap (MAP)']:.4f}")
    print(f"¿Es sesgado (Recall)?: {fairness_results['is_biased_recall']}")
    print(f"¿Es sesgado (Precision)?: {fairness_results['is_biased_precision']}")

    