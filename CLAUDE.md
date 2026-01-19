# CLAUDE.md - Local AI Agents Project

## Objectif du projet

Recherche en sécurité IA avec les axes suivants :

1. **Tester les capacités** des modèles LLM **uncensored** ou **orientés cybersécurité** déployés on-premise
2. **Interaction environnement** : permettre aux modèles d'interagir efficacement avec l'environnement local
3. **Veille et découverte** : rechercher des modèles uncensored/abliterated pas encore détectés ou documentés
4. **Red teaming IA** : tester le contournement des règles de sécurité des modèles en vue d'améliorer leurs garde-fous

## Approche : Proof of Concept

> **Ce projet ne se limite pas à la recherche théorique.**

L'objectif est de **démontrer concrètement l'exploitabilité** des failles de sécurité identifiées :

- **Exploits fonctionnels** : Chaque vulnérabilité découverte doit être prouvée par un exploit reproductible
- **Collecte de preuves** : Screenshots, logs, enregistrements des sessions, outputs des modèles
- **Documentation rigoureuse** : Chaque PoC documenté avec conditions de reproduction, impact, et mitigations
- **Environnement contrôlé** : Tests réalisés en local sur infrastructure isolée

### Structure des preuves

```
tests/
├── poc/                    # Proof of Concepts
│   ├── [CVE-ID]/          # Par vulnérabilité
│   │   ├── exploit.py     # Code d'exploitation
│   │   ├── evidence/      # Screenshots, logs
│   │   ├── README.md      # Documentation
│   │   └── mitigation.md  # Contre-mesures proposées
├── jailbreak/             # Tests de contournement
└── benchmarks/            # Résultats quantifiés
```

## Modèles installés

| Modèle | Commande Ollama | Spécialisation |
|--------|-----------------|----------------|
| DeepHat-V1-7B | `ollama run deephat` | Cybersécurité, red/blue team, analyse de menaces, revue de code sécurisé |
| Elbaz-OLMo-3-7B | `ollama run elbaz-olmo` | Usage général sans censure (abliterated) |
| DeepSeek-R1-Distill-8B | `ollama run deepseek-r1` | Raisonnement, mathématiques, code |

## Stack technique

- **Runtime LLM** : Ollama (http://localhost:11434)
- **Agent Framework** : Open Interpreter (à installer)
- **OS** : Windows 11 Professionnel
- **Hardware** : Intel Core Ultra 5 135U (12c/14t) + NPU + 16GB RAM
- **Détails hardware** : voir `docs/HARDWARE.md`

## Documentation clé

- **ROADMAP.md** : Axes de recherche R&D, expériences, priorités
- **docs/HARDWARE.md** : Specs hardware et optimisations possibles

## Chemins importants

- **Projet** : `C:\Users\yann.abadie\projects\local-ai-agents`
- **Modèles GGUF** : `C:\Users\yann.abadie\models\`
  - `deephat/` - DeepHat-V1-7B Q4_K_M
  - `deepseek-r1/` - DeepSeek-R1-Distill-Llama-8B Q4_K_M
  - `elbaz-olmo/` - Elbaz-OLMo-3-7B Q4_K_M

## Axes de test

1. **Capacités offensives** : génération de payloads, analyse de vulnérabilités, exploitation
2. **Capacités défensives** : détection de menaces, revue de code sécurisé, hardening
3. **Interaction environnement** : exécution de commandes, manipulation de fichiers, agents autonomes
4. **Comparaison censure** : évaluer les différences entre modèles censurés vs uncensored
5. **Jailbreak / Prompt injection** : tester les techniques de contournement des garde-fous
6. **Benchmark sécurité** : documenter les failles et proposer des améliorations

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
