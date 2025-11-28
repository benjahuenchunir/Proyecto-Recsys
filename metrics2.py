import numpy as np
import pandas as pd
import json
from metrics import precision_at_k, ndcg_at_k, novelty, diversity, recall_at_k, average_precision_at_k
import tqdm

def evaluate_model(n, user_groups, user_items_test: dict, item_popularity: dict, item_categories: dict, get_recommendations, ratio_threshold=0.9):
    """
    Evalúa el modelo calculando métricas globales y de fairness.
    
    Args:
        n (int): El 'k' para las métricas @k.
        user_groups (dict): {user_id: 'group_label'}
        ratio_threshold (float): Umbral de ratio (e.g., 0.9). Si el ratio del peor/mejor es < 0.9, se considera sesgado.
    """
    
    group_scores = {
        'recall': {},  
        'precision': {},
        'category_counts': {} # Nuevo: Contadores de categorías recomendadas por grupo
    }
    
    all_recalls = []
    all_aps = []
    all_ndcgs = []
    all_novelties = []
    all_diversities = []
    all_precisions = []
    
    for user_id in tqdm.tqdm(user_items_test.keys(), desc="Evaluando usuarios"):
        
        # 1. Obtener datos del usuario
        if user_id not in user_groups:
            continue
            
        group = user_groups[user_id]
        truth_items = user_items_test.get(user_id, set())
        
        # 2. Obtener recomendaciones y vector de relevancia
        rec = get_recommendations(user_id, n)
        
        # 3. Almacenar para métricas de fairness por grupo
        if group not in group_scores['recall']:
            group_scores['recall'][group] = []
            group_scores['precision'][group] = []
            group_scores['category_counts'][group] = {}

        # Métrica de Rendimiento
        rel_vector = np.isin(rec, truth_items, assume_unique=True).astype(int)
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

        # 5. Almacenar métricas de fairness (Rendimiento)
        group_scores['recall'][group].append(user_recall)
        group_scores['precision'][group].append(user_precision)
        
        # 6. Almacenar métricas de fairness (Categoría/Estereotipo)
        # Contar categorías de los items recomendados
        for item_id in rec:
            category = item_categories.get(item_id, 'UNKNOWN')
            group_scores['category_counts'][group][category] = group_scores['category_counts'][group].get(category, 0) + 1


    # --- 7. Calcular Métricas Promedio (Globales) ---
    metrics_global = {
        'mean_recall': np.mean(all_recalls) if all_recalls else 0.0,
        'mean_precision': np.mean(all_precisions) if all_precisions else 0.0,
        'mean_ap (MAP)': np.mean(all_aps) if all_aps else 0.0,
        'mean_ndcg': np.mean(all_ndcgs) if all_ndcgs else 0.0,
        'mean_novelty': np.mean(all_novelties) if all_novelties else 0.0,
        'mean_diversity': np.mean(all_diversities) if all_diversities else 0.0,
        'num_users_evaluated': len(all_recalls)
    }
    
    # --- 8. Calcular Métricas de Fairness (Disparidad) ---
    
    fairness_report = {
        'ratio_threshold': ratio_threshold,
        'is_biased_recall_ratio': 0,
        'is_biased_precision_ratio': 0,
        'group_averages': {},
        'disparity_reports': [],
        'category_disparity_report': {}
    }

    avg_recall_group = {g: np.mean(s) for g, s in group_scores['recall'].items() if s}
    avg_precision_group = {g: np.mean(s) for g, s in group_scores['precision'].items() if s}
    
    # Pre-cálculo para el reporte de categorías
    group_category_proportions = {}
    for group, counts in group_scores['category_counts'].items():
        total_recs = sum(counts.values())
        if total_recs > 0:
            group_category_proportions[group] = {
                cat: count / total_recs for cat, count in counts.items()
            }
    
    fairness_report['group_averages'] = {
        g: {
            'recall (Cobertura)': avg_recall_group.get(g, 0.0),
            'precision (Tasa Aceptación)': avg_precision_group.get(g, 0.0),
            'count': len(group_scores['recall'].get(g, []))
        } for g in avg_recall_group.keys()
    }

    # Comparar pares de grupos (Disparidad Proporcional)
    group_names = list(avg_recall_group.keys())
    for i in range(len(group_names)):
        for j in range(i + 1, len(group_names)):
            g_a = group_names[i]
            g_b = group_names[j]
            
            # --- Disparidad de Rendimiento (Ratio) ---
            
            r_a, r_b = avg_recall_group[g_a], avg_recall_group[g_b]
            # Evitar división por cero
            if r_a > 0 or r_b > 0:
                recall_ratio = min(r_a, r_b) / max(r_a, r_b) if max(r_a, r_b) > 0 else 1.0
                is_biased_recall = 1 if recall_ratio < ratio_threshold else 0
            else: # Ambos son cero
                recall_ratio = 1.0
                is_biased_recall = 0

            p_a, p_b = avg_precision_group[g_a], avg_precision_group[g_b]
            if p_a > 0 or p_b > 0:
                precision_ratio = min(p_a, p_b) / max(p_a, p_b) if max(p_a, p_b) > 0 else 1.0
                is_biased_precision = 1 if precision_ratio < ratio_threshold else 0
            else:
                precision_ratio = 1.0
                is_biased_precision = 0

            pair_report = {
                'pair': (g_a, g_b),
                'recall_ratio': recall_ratio,
                'precision_ratio': precision_ratio,
                'biased_recall (Rendimiento)': is_biased_recall,
                'biased_precision (Rendimiento)': is_biased_precision
            }
            fairness_report['disparity_reports'].append(pair_report)
            
            if is_biased_recall:
                fairness_report['is_biased_recall_ratio'] = 1
            if is_biased_precision:
                fairness_report['is_biased_precision_ratio'] = 1
            
            # --- Disparidad Categórica (Estereotipo) ---
            
            category_disparities = {}
            # Obtener todas las categorías vistas por ambos grupos
            all_categories = set(group_category_proportions.get(g_a, {}).keys()) | set(group_category_proportions.get(g_b, {}).keys())
            
            for cat in all_categories:
                prop_a = group_category_proportions.get(g_a, {}).get(cat, 0.0)
                prop_b = group_category_proportions.get(g_b, {}).get(cat, 0.0)
                # La disparidad es la diferencia absoluta en la proporción de recomendaciones de esa categoría
                category_disparities[cat] = abs(prop_a - prop_b)
                
            fairness_report['category_disparity_report'][f'{g_a}_vs_{g_b}'] = category_disparities


    return metrics_global, fairness_report

# La función get_metrics se mantiene igual, ya que solo llama a evaluate_model
def get_metrics(user_items_test, item_popularity, item_categories, get_recommendations, k=10, ratio_threshold=0.9):
    profiles = pd.read_csv('clean_data/profiles.csv')

    user_groups_map = {}
    # user_groups_map id a sexo
    for row in profiles.itertuples():
        user_id = row.user_id
        user_groups_map[user_id] = row.gender

    # Llamar a evaluate_model con el nuevo parámetro ratio_threshold
    global_metrics, fairness_results = evaluate_model(k, user_groups_map, user_items_test, item_popularity, item_categories, get_recommendations, ratio_threshold=ratio_threshold)

    print("--- Métricas Globales de Evaluación ---")
    print(json.dumps(global_metrics, indent=2))

    print("\n--- Reporte de Fairness (Disparidad de Grupo) ---")
    print(json.dumps(fairness_results, indent=2))

    print(f"\nRecall Global: {global_metrics['mean_recall']:.4f}")
    print(f"MAP Global: {global_metrics['mean_ap (MAP)']:.4f}")
    print(f"¿Es sesgado (Recall Ratio < {ratio_threshold})?: {fairness_results['is_biased_recall_ratio']}")
    print(f"¿Es sesgado (Precision Ratio < {ratio_threshold})?: {fairness_results['is_biased_precision_ratio']}")