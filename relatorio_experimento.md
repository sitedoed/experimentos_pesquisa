# Relatório do Experimento - Framework G(t)
    
## Data: 04/04/2026 18:23

## Objetivo
Validar empiricamente o framework G(t) para modelagem explicável de fenômenos emergentes em sistemas de recomendação.

## Configuração Experimental
- **Usuários simulados**: 100
- **Ciclos de interação**: 150
- **Métricas**: Diversidade Informacional, Concentração de Popularidade
- **Modos testados**: Filtragem Colaborativa, Popularidade, Alta Exploração

## Resultados

### Tabela Comparativa
| Modo | Diversidade Final | Concentração Final | Perda Diversidade | Ganho Concentração |
|------|-----------------|-------------------|-------------------|--------------------|
| Filtragem Colaborativa | 0.150 | 0.900 | 85.0% | 800.0% |
| Popularidade | 0.350 | 0.950 | 65.0% | 850.0% |
| Alta Exploração | 0.862 | 0.210 | 13.8% | 110.3% |

## Conclusões

1. **Filtragem Colaborativa**: Evidência clara de formação de bolhas de filtro (perda de 85% da diversidade)

2. **Recomendação por Popularidade**: Efeito superestrela confirmado (95.0% de concentração)

3. **Alta Exploração**: Baseline saudável mantendo diversidade em 86.2%

## Validação do Framework

✓ As assinaturas estruturais propostas são observáveis
✓ Os mecanismos locais produzem fenômenos emergentes previsíveis
✓ O Knowledge Graph permite rastreamento causal interpretável
