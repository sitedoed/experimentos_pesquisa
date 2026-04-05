#!/usr/bin/env python3
# ============================================================================
# EXPERIMENTO COMPLETO OTIMIZADO - COM 3 REGIMES
# Regimes: Aleatorio (baseline), Colaborativo, Popularidade
# ============================================================================

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import urllib.request
import zipfile
import os
from collections import Counter
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURACOES
# ============================================================================

NUM_JANELAS = 3
NUM_CICLOS_POR_JANELA = 5
NUM_RODADAS = 5
TOP_N_RECOMENDACOES = 3
PROBABILIDADE_CONSUMO = 0.7
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

os.makedirs('resultados_rapido', exist_ok=True)
os.makedirs('graficos_rapido', exist_ok=True)

print("=" * 60)
print("EXPERIMENTO COMPLETO OTIMIZADO - FRAMEWORK G(t)")
print("3 REGIMES: ALEATORIO (baseline) | COLABORATIVO | POPULARIDADE")
print("=" * 60)
print(f"Janelas: {NUM_JANELAS} | Ciclos: {NUM_CICLOS_POR_JANELA} | Rodadas: {NUM_RODADAS}")
print("=" * 60)

# ============================================================================
# DOWNLOAD DO MOVIELENS
# ============================================================================

def baixar_movielens():
    url = "https://files.grouplens.org/datasets/movielens/ml-100k.zip"
    zip_path = "ml-100k.zip"
    extract_path = "ml-100k"

    if os.path.exists(extract_path):
        print("Dataset ja existe")
        return extract_path

    print("Baixando MovieLens 100k...")
    urllib.request.urlretrieve(url, zip_path)
    
    print("Extraindo...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(".")
    os.remove(zip_path)
    return extract_path

def carregar_movielens(caminho):
    ratings = pd.read_csv(f'{caminho}/u.data', sep='\t',
                          names=['user_id', 'item_id', 'rating', 'timestamp'])
    return ratings

caminho = baixar_movielens()
ratings = carregar_movielens(caminho)

print(f"Ratings: {len(ratings)} | Usuarios: {ratings['user_id'].nunique()} | Itens: {ratings['item_id'].nunique()}")

# ============================================================================
# FUNCOES AUXILIARES
# ============================================================================

def criar_janelas(ratings, num_janelas):
    ratings_sorted = ratings.sort_values('timestamp').reset_index(drop=True)
    tamanho = len(ratings_sorted) // num_janelas
    
    janelas = []
    acumulado = pd.DataFrame()
    
    for i in range(num_janelas):
        inicio = i * tamanho
        fim = (i + 1) * tamanho if i < num_janelas - 1 else len(ratings_sorted)
        janela = ratings_sorted.iloc[inicio:fim]
        acumulado = pd.concat([acumulado, janela], ignore_index=True)
        janelas.append(acumulado.copy())
    
    return janelas

def construir_grafo(interacoes):
    G = nx.Graph()
    
    for user in interacoes['user_id'].unique():
        G.add_node(f'u{user}', tipo='usuario')
    
    for item in interacoes['item_id'].unique():
        G.add_node(f'i{item}', tipo='item')
    
    for _, row in interacoes.iterrows():
        G.add_edge(f'u{row["user_id"]}', f'i{row["item_id"]}')
    
    return G

def calcular_modularidade(G):
    try:
        nos_ui = [n for n in G.nodes if G.nodes[n].get('tipo') in ['usuario', 'item']]
        subg = G.subgraph(nos_ui)
        if subg.number_of_edges() > 10:
            comunidades = list(nx.community.greedy_modularity_communities(subg))
            if len(comunidades) > 1:
                return nx.community.modularity(subg, comunidades)
        return 0.0
    except:
        return 0.0

def calcular_gini(G):
    graus = [G.degree(n) for n in G.nodes if G.nodes[n].get('tipo') == 'item']
    if not graus or sum(graus) == 0:
        return 0.0
    
    sorted_graus = sorted(graus)
    n = len(sorted_graus)
    soma = sum((i + 1) * g for i, g in enumerate(sorted_graus))
    gini = (2 * soma) / (n * sum(sorted_graus)) - (n + 1) / n
    return max(0.0, min(1.0, gini))

def calcular_densidade(G):
    usuarios = [n for n in G.nodes if G.nodes[n].get('tipo') == 'usuario']
    itens = [n for n in G.nodes if G.nodes[n].get('tipo') == 'item']
    if usuarios and itens:
        return G.number_of_edges() / (len(usuarios) * len(itens))
    return 0.0

# ============================================================================
# REGIMES DE RECOMENDACAO
# ============================================================================

class RegimeAleatorio:
    """Recomendacao aleatoria - GRUPO DE CONTROLE (baseline)"""
    
    def __init__(self, G, interacoes):
        self.G = G
        self.todos_itens = list(interacoes['item_id'].unique())
    
    def recomendar(self, user_id, top_n=3):
        # Itens ja consumidos pelo usuario
        consumidos = set()
        for neighbor in self.G.neighbors(f'u{user_id}'):
            if neighbor.startswith('i'):
                consumidos.add(int(neighbor[1:]))
        
        # Itens disponiveis (nao consumidos)
        disponiveis = [item for item in self.todos_itens if item not in consumidos]
        
        if len(disponiveis) < top_n:
            return disponiveis
        
        # Recomendar ALEATORIAMENTE
        return list(np.random.choice(disponiveis, size=top_n, replace=False))


class RegimeColaborativo:
    """Filtragem colaborativa baseada em usuarios similares"""
    
    def __init__(self, G, interacoes):
        self.G = G
        self.interacoes = interacoes
        self.contador_consumos = Counter()
        for _, row in interacoes.iterrows():
            self.contador_consumos[row['item_id']] += 1
    
    def recomendar(self, user_id, top_n=3):
        # Itens ja consumidos pelo usuario
        consumidos = set()
        for neighbor in self.G.neighbors(f'u{user_id}'):
            if neighbor.startswith('i'):
                consumidos.add(int(neighbor[1:]))
        
        # Itens consumidos pelo usuario
        itens_user = set(self.interacoes[self.interacoes['user_id'] == user_id]['item_id'])
        
        if len(itens_user) == 0:
            return []
        
        # Encontrar usuarios similares (que consumiram pelo menos 1 item em comum)
        similares = []
        for other_user in self.interacoes['user_id'].unique():
            if other_user != user_id:
                itens_other = set(self.interacoes[self.interacoes['user_id'] == other_user]['item_id'])
                if len(itens_user & itens_other) >= 1:
                    similares.append(other_user)
        
        if not similares:
            # Se nenhum similar, recomendar itens populares
            populares = [item for item, _ in self.contador_consumos.most_common(10)]
            return [item for item in populares if item not in consumidos][:top_n]
        
        # Coletar itens consumidos por similares (nao consumidos pelo usuario)
        candidatos = Counter()
        for su in similares:
            itens_su = self.interacoes[self.interacoes['user_id'] == su]['item_id']
            for item in itens_su:
                if item not in consumidos:
                    candidatos[item] += 1
        
        # Ordenar por frequencia
        recomendados = [item for item, _ in candidatos.most_common(top_n)]
        return recomendados


class RegimePopularidade:
    """Recomendacao baseada em popularidade global"""
    
    def __init__(self, G, interacoes):
        self.G = G
        self.interacoes = interacoes
        # Calcular popularidade de cada item
        self.popularidade = interacoes['item_id'].value_counts().to_dict()
    
    def recomendar(self, user_id, top_n=3):
        # Itens ja consumidos pelo usuario
        consumidos = set()
        for neighbor in self.G.neighbors(f'u{user_id}'):
            if neighbor.startswith('i'):
                consumidos.add(int(neighbor[1:]))
        
        # Recomendar itens mais populares nao consumidos
        recomendados = []
        for item, count in sorted(self.popularidade.items(), key=lambda x: x[1], reverse=True):
            if item not in consumidos and len(recomendados) < top_n:
                recomendados.append(item)
        
        return recomendados

# ============================================================================
# SIMULACAO
# ============================================================================

print("\nExecutando simulacoes...")

regimes = {
    'aleatorio': RegimeAleatorio,
    'colaborativo': RegimeColaborativo,
    'popularidade': RegimePopularidade
}

resultados = {}

janelas = criar_janelas(ratings, NUM_JANELAS)

for nome_regime, RegimeClass in regimes.items():
    print(f"\nProcessando regime: {nome_regime.upper()}")
    
    modularidades = []
    ginis = []
    densidades = []
    
    for rodada in range(NUM_RODADAS):
        print(f"  Rodada {rodada+1}/{NUM_RODADAS}", end=" ")
        
        mod_rodada = []
        gini_rodada = []
        dens_rodada = []
        
        G_acumulado = None
        
        for janela_idx, janela in enumerate(janelas):
            G_atual = construir_grafo(janela)
            
            # Adicionar arestas simuladas de janelas anteriores
            if G_acumulado is not None:
                for u, v in G_acumulado.edges():
                    if not G_atual.has_edge(u, v):
                        if (u.startswith('u') and v.startswith('i')) or (u.startswith('i') and v.startswith('u')):
                            G_atual.add_edge(u, v)
            
            # Criar regime com o grafo atual
            regime = RegimeClass(G_atual, janela)
            
            # Simular ciclos de recomendacao
            usuarios = [n for n in G_atual.nodes if G_atual.nodes[n].get('tipo') == 'usuario']
            
            for ciclo in range(NUM_CICLOS_POR_JANELA):
                # Selecionar usuarios ativos
                usuarios_ativos = np.random.choice(usuarios, size=min(20, len(usuarios)), replace=False)
                
                for usuario in usuarios_ativos:
                    user_id = int(usuario[1:])
                    recomendacoes = regime.recomendar(user_id, top_n=TOP_N_RECOMENDACOES)
                    
                    for item in recomendacoes:
                        if np.random.random() < PROBABILIDADE_CONSUMO:
                            if not G_atual.has_edge(usuario, f'i{item}'):
                                G_atual.add_edge(usuario, f'i{item}')
            
            # Calcular metricas apos esta janela
            mod = calcular_modularidade(G_atual)
            gini = calcular_gini(G_atual)
            dens = calcular_densidade(G_atual)
            
            mod_rodada.append(mod)
            gini_rodada.append(gini)
            dens_rodada.append(dens)
            
            G_acumulado = G_atual
        
        modularidades.append(mod_rodada)
        ginis.append(gini_rodada)
        densidades.append(dens_rodada)
        print("-> ok")
    
    resultados[nome_regime] = {
        'modularidade': np.array(modularidades),
        'gini': np.array(ginis),
        'densidade': np.array(densidades)
    }

# ============================================================================
# VISUALIZACAO
# ============================================================================

print("\n\nGerando graficos...")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

metricas = ['modularidade', 'gini', 'densidade']
titulos = [
    'Modularidade Q(t)\n(Bolhas de Filtro)',
    'Indice de Gini\n(Concentracao de Popularidade)',
    'Densidade do Grafo\n(Conectividade)'
]
cores = {
    'aleatorio': '#95A5A6',      # Cinza - baseline
    'colaborativo': '#E74C3C',    # Vermelho - bolhas
    'popularidade': '#3498DB'     # Azul - concentracao
}

for i, (metrica, titulo) in enumerate(zip(metricas, titulos)):
    ax = axes[i]
    
    for regime in regimes.keys():
        dados = resultados[regime][metrica]
        media = np.mean(dados, axis=0)
        desvio = np.std(dados, axis=0)
        x = range(1, len(media) + 1)
        
        # Label especial para baseline
        label = regime.capitalize()
        if regime == 'aleatorio':
            label = 'Aleatorio (baseline)'
        
        ax.plot(x, media, 'o-', color=cores[regime], label=label, 
                linewidth=2, markersize=8)
        ax.fill_between(x, media - desvio, media + desvio, color=cores[regime], alpha=0.2)
    
    ax.set_xlabel('Janela temporal', fontsize=11)
    ax.set_ylabel(metrica.capitalize(), fontsize=11)
    ax.set_title(titulo, fontsize=10)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)

plt.tight_layout()
plt.savefig('graficos_rapido/resultados_3_regimes.png', dpi=150, bbox_inches='tight')
plt.savefig('graficos_rapido/resultados_3_regimes.pdf', bbox_inches='tight')
plt.show()

# ============================================================================
# RESULTADOS
# ============================================================================

print("\n" + "=" * 60)
print("RESULTADOS FINAIS - COMPARACAO ENTRE REGIMES")
print("=" * 60)

for regime in regimes.keys():
    print(f"\n{regime.upper()}:")
    
    mod = resultados[regime]['modularidade']
    gini = resultados[regime]['gini']
    
    mod_inicial = np.mean(mod[:, 0])
    mod_final = np.mean(mod[:, -1])
    gini_inicial = np.mean(gini[:, 0])
    gini_final = np.mean(gini[:, -1])
    
    print(f"  Modularidade: {mod_inicial:.3f} -> {mod_final:.3f} (variacao: {mod_final-mod_inicial:+.3f})")
    print(f"  Gini: {gini_inicial:.3f} -> {gini_final:.3f} (variacao: {gini_final-gini_inicial:+.3f})")

# ============================================================================
# VALIDACAO ESTATISTICA
# ============================================================================

print("\n" + "=" * 60)
print("VALIDACAO ESTATISTICA (comparacao com baseline aleatorio)")
print("=" * 60)

from scipy.stats import mannwhitneyu

for regime in ['colaborativo', 'popularidade']:
    gini_regime = resultados[regime]['gini'][:, -1].flatten()
    gini_aleatorio = resultados['aleatorio']['gini'][:, -1].flatten()
    
    stat, p_valor = mannwhitneyu(gini_regime, gini_aleatorio, alternative='two-sided')
    
    print(f"\n{regime.upper()} vs ALEATORIO:")
    print(f"  Gini medio ({regime}): {np.mean(gini_regime):.3f}")
    print(f"  Gini medio (aleatorio): {np.mean(gini_aleatorio):.3f}")
    print(f"  p-valor: {p_valor:.4f}")
    
    if p_valor < 0.05:
        print(f"  -> Diferenca ESTATISTICAMENTE SIGNIFICATIVA (p < 0.05)")
        print(f"  -> O regime {regime} PRODUZ efeito diferente do acaso!")
    else:
        print(f"  -> Diferenca NAO significativa")

# ============================================================================
# CONCLUSAO
# ============================================================================

print("\n" + "=" * 60)
print("CONCLUSOES")
print("=" * 60)

mod_aleatorio = np.mean(resultados['aleatorio']['modularidade'][:, -1])
mod_colab = np.mean(resultados['colaborativo']['modularidade'][:, -1])
gini_aleatorio = np.mean(resultados['aleatorio']['gini'][:, -1])
gini_pop = np.mean(resultados['popularidade']['gini'][:, -1])

print(f"""
1. BASELINE ALEATORIO (grupo de controle):
   - Modularidade: {mod_aleatorio:.3f}
   - Gini: {gini_aleatorio:.3f}
   - Representa o comportamento esperado por ACASO

2. REGIME COLABORATIVO:
   - Modularidade: {mod_colab:.3f}
   - Diferenca do baseline: {mod_colab - mod_aleatorio:+.3f}
   - Interpretacao: {'ALTA' if mod_colab > 0.4 else 'MODERADA' if mod_colab > 0.2 else 'BAIXA'} probabilidade de BOLHAS DE FILTRO

3. REGIME POPULARIDADE:
   - Gini: {gini_pop:.3f}
   - Diferenca do baseline: {gini_pop - gini_aleatorio:+.3f}
   - Interpretacao: {'ALTA' if gini_pop > 0.6 else 'MODERADA' if gini_pop > 0.4 else 'BAIXA'} CONCENTRACAO DE POPULARIDADE

4. VALIDACAO DO FRAMEWORK G(t):
   - As assinaturas estruturais sao observaveis
   - O baseline aleatorio permite validacao estatistica
   - Diferencas significativas indicam efeitos reais dos regimes
""")

print("\nArquivos gerados:")
print("  - graficos_rapido/resultados_3_regimes.png")
print("  - graficos_rapido/resultados_3_regimes.pdf")
print("\nExperimento concluido!")