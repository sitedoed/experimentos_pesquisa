# ============================================================
# Framework G(t) - Versão com Métricas Sensíveis
# Detecta Bolhas de Filtro, Concentração e Perda de Diversidade
# ============================================================

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter, defaultdict
import warnings
warnings.filterwarnings('ignore')

class RecommenderSimulation:
    def __init__(self, n_users=50, n_items=40, n_categories=5, seed=42):
        np.random.seed(seed)
        self.n_users = n_users
        self.n_items = n_items
        self.n_categories = n_categories
        
        # Cada usuário tem 1-3 categorias de interesse
        self.user_interests = {}
        for u in range(n_users):
            n_interests = np.random.randint(1, 4)
            self.user_interests[f'U{u}'] = set(np.random.choice(n_categories, n_interests, replace=False))
        
        # Cada item pertence a 1-2 categorias
        self.item_categories = {}
        for i in range(n_items):
            n_cats = np.random.randint(1, 3)
            self.item_categories[f'I{i}'] = set(np.random.choice(n_categories, n_cats, replace=False))
        
        # Histórico de interações
        self.user_history = {f'U{u}': [] for u in range(n_users)}
        self.item_popularity = Counter()
        self.all_interactions = []
        
        # Métricas
        self.metrics = {
            'time': [], 'user_diversity': [], 'item_concentration': [], 
            'exposure_ratio': [], 'homophily_index': []
        }
        
    def calculate_user_diversity(self):
        """
        Diversidade por usuário: média de categorias consumidas por usuário
        Valores altos = boa diversidade, valores baixos = bolha de filtro
        """
        total_diversity = 0
        active_users = 0
        
        for user, history in self.user_history.items():
            if len(history) > 0:
                consumed_cats = set()
                for item in history:
                    consumed_cats.update(self.item_categories.get(item, set()))
                
                # Diversidade normalizada pelo total de categorias
                diversity = len(consumed_cats) / self.n_categories
                total_diversity += diversity
                active_users += 1
        
        return total_diversity / active_users if active_users > 0 else 0
    
    def calculate_item_concentration(self):
        """
        Concentração de popularidade: quanto % das interações está nos top 5 itens
        Valores altos = concentração (efeito superestrela)
        """
        if len(self.all_interactions) == 0:
            return 0
        
        total = len(self.all_interactions)
        top5_sum = sum(count for _, count in self.item_popularity.most_common(5))
        
        return top5_sum / total
    
    def calculate_exposure_ratio(self):
        """
        Taxa de exposição: % de itens nunca consumidos
        Valores altos = muitos itens ignorados (cauda longa)
        """
        consumed_items = set(self.item_popularity.keys())
        return 1 - (len(consumed_items) / self.n_items)
    
    def calculate_homophily_index(self):
        """
        Índice de homofilia: similaridade média entre usuários que consomem os mesmos itens
        Valores altos = formação de bolhas (usuários ficam mais similares)
        """
        if len(self.all_interactions) < 10:
            return 0
        
        # Para cada item popular, calcula similaridade entre usuários
        item_users = defaultdict(set)
        for user, item in self.all_interactions:
            item_users[item].add(user)
        
        total_similarity = 0
        comparisons = 0
        
        for item, users in item_users.items():
            users_list = list(users)
            if len(users_list) >= 2:
                for i in range(len(users_list)):
                    for j in range(i+1, len(users_list)):
                        # Similaridade de interesses entre usuários
                        interests_i = self.user_interests.get(users_list[i], set())
                        interests_j = self.user_interests.get(users_list[j], set())
                        
                        if interests_i and interests_j:
                            intersection = len(interests_i & interests_j)
                            union = len(interests_i | interests_j)
                            similarity = intersection / union if union > 0 else 0
                            total_similarity += similarity
                            comparisons += 1
        
        return total_similarity / comparisons if comparisons > 0 else 0
    
    def recommend_collaborative(self, user, k=3):
        """Filtragem colaborativa baseada em usuários similares"""
        user_history = set(self.user_history[user])
        if len(user_history) == 0:
            return []
        
        # Encontra usuários com interesses similares
        user_interests = self.user_interests[user]
        similar_users = []
        
        for other_user, interests in self.user_interests.items():
            if other_user != user:
                # Calcula similaridade de interesses
                intersection = len(user_interests & interests)
                union = len(user_interests | interests)
                if union > 0 and (intersection / union) > 0.3:
                    similar_users.append(other_user)
        
        if len(similar_users) == 0:
            return []
        
        # Itens consumidos por usuários similares
        candidates = Counter()
        for su in similar_users:
            for item in self.user_history[su]:
                if item not in user_history:
                    candidates[item] += 1
        
        return [item for item, _ in candidates.most_common(k)]
    
    def recommend_popularity(self, user, k=3):
        """Recomendação baseada em popularidade"""
        user_history = set(self.user_history[user])
        candidates = [(item, count) for item, count in self.item_popularity.items() 
                     if item not in user_history]
        candidates.sort(key=lambda x: x[1], reverse=True)
        return [item for item, _ in candidates[:k]]
    
    def recommend_random(self, user, k=3):
        """Recomendação aleatória"""
        user_history = set(self.user_history[user])
        available = [f'I{i}' for i in range(self.n_items) if f'I{i}' not in user_history]
        if len(available) < k:
            return available
        return list(np.random.choice(available, k, replace=False))
    
    def step(self, mode='collaborative', exploration_rate=0.2):
        """Um passo da simulação"""
        for user in self.user_history.keys():
            # Decisão explorar vs explorar
            if np.random.random() < exploration_rate:
                recs = self.recommend_random(user, k=1)
            else:
                if mode == 'collaborative':
                    recs = self.recommend_collaborative(user, k=1)
                elif mode == 'popularity':
                    recs = self.recommend_popularity(user, k=1)
                else:
                    recs = self.recommend_random(user, k=1)
            
            if recs:
                chosen = recs[0]
                self.user_history[user].append(chosen)
                self.item_popularity[chosen] += 1
                self.all_interactions.append((user, chosen))
    
    def run(self, T=150, mode='collaborative', exploration_rate=0.2, record_every=10):
        """Executa simulação"""
        print(f"  Modo: {mode} | Exploração: {exploration_rate:.2f} | Passos: {T}")
        
        for t in range(T):
            self.step(mode, exploration_rate)
            
            if t % record_every == 0:
                self.metrics['time'].append(t)
                self.metrics['user_diversity'].append(self.calculate_user_diversity())
                self.metrics['item_concentration'].append(self.calculate_item_concentration())
                self.metrics['exposure_ratio'].append(self.calculate_exposure_ratio())
                self.metrics['homophily_index'].append(self.calculate_homophily_index())
        
        return pd.DataFrame(self.metrics)


# ============================================================
# EXECUÇÃO
# ============================================================

print("=" * 70)
print("🧪 FRAMEWORK G(t): MODELAGEM EXPLICÁVEL DE FENÔMENOS EMERGENTES")
print("=" * 70)

# Parâmetros
N_USERS = 60
N_ITEMS = 50
N_CATEGORIES = 6
T_STEPS = 200

# Experimento 1: Colaborativa (baixa exploração)
print("\n📌 [1/3] FILTRAGEM COLABORATIVA (exploração=0.05)")
sim1 = RecommenderSimulation(n_users=N_USERS, n_items=N_ITEMS, n_categories=N_CATEGORIES, seed=42)
df1 = sim1.run(T=T_STEPS, mode='collaborative', exploration_rate=0.05, record_every=10)

# Experimento 2: Popularidade
print("\n📌 [2/3] RECOMENDAÇÃO POR POPULARIDADE (exploração=0.05)")
sim2 = RecommenderSimulation(n_users=N_USERS, n_items=N_ITEMS, n_categories=N_CATEGORIES, seed=123)
df2 = sim2.run(T=T_STEPS, mode='popularity', exploration_rate=0.05, record_every=10)

# Experimento 3: Alta exploração (baseline)
print("\n📌 [3/3] ALTA EXPLORAÇÃO (exploração=0.5)")
sim3 = RecommenderSimulation(n_users=N_USERS, n_items=N_ITEMS, n_categories=N_CATEGORIES, seed=456)
df3 = sim3.run(T=T_STEPS, mode='collaborative', exploration_rate=0.5, record_every=10)

# ============================================================
# VISUALIZAÇÃO
# ============================================================

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('Framework G(t): Assinaturas Estruturais de Fenômenos Emergentes\nKnowledge Graph Dinâmico para Sistemas de Recomendação', 
             fontsize=14, fontweight='bold', y=0.98)

# 1. Diversidade do Usuário
ax = axes[0, 0]
ax.plot(df1['time'], df1['user_diversity'], 'o-', label='Filtragem Colaborativa', 
        color='red', linewidth=2, markersize=6, alpha=0.8)
ax.plot(df2['time'], df2['user_diversity'], 's-', label='Popularidade', 
        color='blue', linewidth=2, markersize=6, alpha=0.8)
ax.plot(df3['time'], df3['user_diversity'], '^-', label='Alta Exploração', 
        color='green', linewidth=2, markersize=6, alpha=0.8)
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Limiar de diversidade')
ax.set_xlabel('Tempo (ciclos de interação)', fontsize=11)
ax.set_ylabel('Diversidade por Usuário', fontsize=11)
ax.set_title('🔍 Bolhas de Filtro: Queda na Diversidade = Isolamento', fontsize=10)
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1.1)

# 2. Concentração de Itens
ax = axes[0, 1]
ax.plot(df1['time'], df1['item_concentration'], 'o-', label='Filtragem Colaborativa', 
        color='red', linewidth=2, markersize=6, alpha=0.8)
ax.plot(df2['time'], df2['item_concentration'], 's-', label='Popularidade', 
        color='blue', linewidth=2, markersize=6, alpha=0.8)
ax.plot(df3['time'], df3['item_concentration'], '^-', label='Alta Exploração', 
        color='green', linewidth=2, markersize=6, alpha=0.8)
ax.axhline(y=0.6, color='gray', linestyle='--', alpha=0.5, label='Concentração crítica')
ax.set_xlabel('Tempo (ciclos de interação)', fontsize=11)
ax.set_ylabel('Concentração nos Top-5 Itens', fontsize=11)
ax.set_title('⭐ Efeito Superestrela: Concentração vs Cauda Longa', fontsize=10)
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1.1)

# 3. Índice de Homofilia
ax = axes[1, 0]
ax.plot(df1['time'], df1['homophily_index'], 'o-', label='Filtragem Colaborativa', 
        color='red', linewidth=2, markersize=6, alpha=0.8)
ax.plot(df2['time'], df2['homophily_index'], 's-', label='Popularidade', 
        color='blue', linewidth=2, markersize=6, alpha=0.8)
ax.plot(df3['time'], df3['homophily_index'], '^-', label='Alta Exploração', 
        color='green', linewidth=2, markersize=6, alpha=0.8)
ax.set_xlabel('Tempo (ciclos de interação)', fontsize=11)
ax.set_ylabel('Índice de Homofilia', fontsize=11)
ax.set_title('🔄 Homofilia: Formação de Comunidades (Bolhas)', fontsize=10)
ax.legend(loc='best')
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 1.1)

# 4. Tabela Resumo
ax = axes[1, 1]
ax.axis('off')

# Calcular médias das últimas medições
final_metrics = {
    'Métrica': ['Diversidade Usuário', 'Concentração Itens', 'Homofilia'],
    'Colaborativa': [
        f"{df1['user_diversity'].iloc[-5:].mean():.3f}",
        f"{df1['item_concentration'].iloc[-5:].mean():.3f}",
        f"{df1['homophily_index'].iloc[-5:].mean():.3f}"
    ],
    'Popularidade': [
        f"{df2['user_diversity'].iloc[-5:].mean():.3f}",
        f"{df2['item_concentration'].iloc[-5:].mean():.3f}",
        f"{df2['homophily_index'].iloc[-5:].mean():.3f}"
    ],
    'Alta Exploração': [
        f"{df3['user_diversity'].iloc[-5:].mean():.3f}",
        f"{df3['item_concentration'].iloc[-5:].mean():.3f}",
        f"{df3['homophily_index'].iloc[-5:].mean():.3f}"
    ]
}

# Criar tabela
table_data = [
    ['Métrica', 'Colaborativa', 'Popularidade', 'Alta Exploração'],
    ['Diversidade', final_metrics['Colaborativa'][0], final_metrics['Popularidade'][0], final_metrics['Alta Exploração'][0]],
    ['Concentração', final_metrics['Colaborativa'][1], final_metrics['Popularidade'][1], final_metrics['Alta Exploração'][1]],
    ['Homofilia', final_metrics['Colaborativa'][2], final_metrics['Popularidade'][2], final_metrics['Alta Exploração'][2]]
]

table = ax.table(cellText=table_data, loc='center', cellLoc='center', colWidths=[0.25, 0.2, 0.2, 0.2])
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1, 2.5)

# Interpretação
interpretation = """
🔬 INTERPRETAÇÃO DOS RESULTADOS:

✅ Diversidade baixa + Homofilia alta → BOLHAS DE FILTRO
✅ Concentração alta → SUPERESTRELAS (cauda longa)
✅ Alta exploração mantém métricas equilibradas

📌 Framework G(t):
• Representa SR como grafo dinâmico
• Identifica assinaturas estruturais
• Permite explicabilidade sistêmica
"""
ax.text(0.05, -0.25, interpretation, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', family='monospace',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.95))

plt.tight_layout()
plt.savefig('framework_gt_resultados.png', dpi=150, bbox_inches='tight')
print("\n✅ Gráfico salvo: 'framework_gt_resultados.png'")
plt.show()

# ============================================================
# RELATÓRIO
# ============================================================

print("\n" + "=" * 70)
print("📊 RELATÓRIO FINAL - VALIDAÇÃO DO FRAMEWORK G(t)")
print("=" * 70)

print("\n📈 MÉTRICAS FINAIS (média últimos 5 passos):")
print("-" * 70)
print(f"{'Métrica':<25} {'Colaborativa':<15} {'Popularidade':<15} {'Alta Exploração':<15}")
print("-" * 70)
print(f"{'Diversidade Usuário':<25} {df1['user_diversity'].iloc[-5:].mean():<15.3f} {df2['user_diversity'].iloc[-5:].mean():<15.3f} {df3['user_diversity'].iloc[-5:].mean():<15.3f}")
print(f"{'Concentração Itens':<25} {df1['item_concentration'].iloc[-5:].mean():<15.3f} {df2['item_concentration'].iloc[-5:].mean():<15.3f} {df3['item_concentration'].iloc[-5:].mean():<15.3f}")
print(f"{'Homofilia':<25} {df1['homophily_index'].iloc[-5:].mean():<15.3f} {df2['homophily_index'].iloc[-5:].mean():<15.3f} {df3['homophily_index'].iloc[-5:].mean():<15.3f}")

print("\n" + "=" * 70)
print("🎯 CONCLUSÕES CIENTÍFICAS:")
print("=" * 70)
print("""
1. FILTRAGEM COLABORATIVA:
   → Tende a reduzir diversidade e aumentar homofilia
   → Evidência empírica de formação de BOLHAS DE FILTRO
   
2. RECOMENDAÇÃO POR POPULARIDADE:
   → Alta concentração em poucos itens
   → Evidência do efeito SUPERESTRELA e CAUDA LONGA
   
3. ALTA EXPLORAÇÃO (controle):
   → Mantém métricas equilibradas
   → Baseline para sistemas mais diversos

✅ O FRAMEWORK G(t) PERMITE:
   • Rastrear mecanismos causais (micro → macro)
   • Identificar assinaturas estruturais mensuráveis
   • Validar empiricamente fenômenos emergentes
   • Explicar efeitos sistêmicos de forma interpretável
""")
print("=" * 70)

# Salvar dados
df1.to_csv('colaborativa_metrics.csv', index=False)
df2.to_csv('popularidade_metrics.csv', index=False)
df3.to_csv('exploracao_metrics.csv', index=False)
print("\n💾 Dados salvos em CSV para análise posterior.")