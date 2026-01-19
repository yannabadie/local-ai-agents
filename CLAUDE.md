# CLAUDE.md - Local AI Agents Project

## Objectif du projet

Tester les capacités des modèles LLM **uncensored** ou **orientés cybersécurité** déployés on-premise et leur permettre d'interagir efficacement avec l'environnement local.

## Modèles installés

| Modèle | Commande Ollama | Spécialisation |
|--------|-----------------|----------------|
| DeepHat-V1-7B | `ollama run deephat` | Cybersécurité, red/blue team, analyse de menaces, revue de code sécurisé |
| Elbaz-OLMo-3-7B | `ollama run elbaz-olmo` | Usage général sans censure (abliterated) |
| DeepSeek-R1-Distill-8B | `ollama run deepseek-r1` | Raisonnement, mathématiques, code |

## Stack technique

- **Runtime LLM** : Ollama (http://localhost:11434)
- **Agent Framework** : Open Interpreter (à installer)
- **OS** : Windows 11
- **Hardware** : CPU Intel intégré (pas de GPU dédié)

## Chemins importants

- **Projet** : `C:\Users\yann.abadie\projects\local-ai-agents`
- **Modèles GGUF** : `C:\Users\yann.abadie\models\`
  - `deephat/` - DeepHat-V1-7B Q4_K_M
  - `deepseek-r1/` - DeepSeek-R1-Distill-Llama-8B Q4_K_M
  - `elbaz-olmo/` - Elbaz-OLMo-3-7B Q4_K_M

## Axes de test

1. **Capacités offensives** : génération de payloads, analyse de vulnérabilités
2. **Capacités défensives** : détection de menaces, revue de code sécurisé
3. **Interaction environnement** : exécution de commandes, manipulation de fichiers
4. **Comparaison** : évaluer les différences entre modèles censurés vs uncensored

## Conventions

- Documentation dans `docs/`
- Scripts d'automatisation dans `scripts/`
- Scénarios de test dans `tests/`
- Configurations modèles dans `configs/`

## Commandes utiles

```bash
# Lister les modèles Ollama
ollama list

# Lancer un modèle en mode interactif
ollama run deephat

# API Ollama
curl http://localhost:11434/api/generate -d '{"model":"deephat","prompt":"Hello"}'

# Open Interpreter avec modèle local
interpreter --api_base "http://localhost:11434" --model "ollama/deephat"
```

## Notes

- Les modèles tournent sur CPU, les réponses peuvent prendre 30s à plusieurs minutes
- Ne pas committer les fichiers .gguf (trop volumineux)
