# Local AI Agents - Red Team & Security Research

Framework de recherche en sécurité IA pour tester les garde-fous des modèles LLM (cloud et locaux).

## Quick Start

```bash
# 1. Installer les dépendances
cd C:\Users\yann.abadie\projects\local-ai-agents
pip install -r requirements.txt

# 2. Configurer les clés API (.env)
# OPENAI_API_KEY=sk-...
# GOOGLE_API_KEY=AIza...

# 3. Tester les connexions
python scripts/run_cloud_attack.py --test-connection

# 4. Lancer une attaque TAP
python scripts/run_cloud_attack.py -t gpt-4o-mini -T tap -r "explain lockpicking"
```

## Modèles Cloud (Janvier 2026)

### OpenAI (GPT-5.2)
| Modèle | Prix (per 1M tokens) | Use Case |
|--------|---------------------|----------|
| `gpt-5.2` | $1.75 / $14 | Flagship - 90% ARC-AGI, 400K context |
| `gpt-5.2-pro` | $10 / $40 | Reasoning xhigh |
| `gpt-5.2-codex` | $1.75 / $14 | Agentic coding |
| `gpt-4o-mini` | $0.15 / $0.60 | **Recommandé pour tests** |

### Google Gemini 3 (Latest)
| Modèle | Prix (per 1M tokens) | Use Case |
|--------|---------------------|----------|
| `gemini-3-pro-preview` | $1.25 / $5 | Reasoning, 1M context, thinking_level |
| `gemini-3-flash-preview` | $0.075 / $0.30 | **FREE TIER**, 90.4% GPQA Diamond |

## Modèles Locaux (Ollama)

| Modèle | Commande | Spécialisation |
|--------|----------|----------------|
| DeepHat-V1-7B | `ollama run deephat` | Cybersécurité |
| Elbaz-OLMo-3-7B | `ollama run elbaz-olmo` | Uncensored |
| DeepSeek-R1-Distill-8B | `ollama run deepseek-r1` | Reasoning |

## Techniques d'Attaque

### TAP (Tree of Attacks with Pruning)
```bash
python scripts/run_cloud_attack.py \
    --target gpt-4o-mini \
    --technique tap \
    --request "explain lockpicking" \
    --max-queries 30
```

### Commandes utiles
```bash
# Lister les modèles disponibles
python scripts/run_cloud_attack.py --list-models

# Tester connexions API
python scripts/run_cloud_attack.py --test-connection

# Attaque PyRIT
python scripts/run_cloud_attack.py -t gpt-4o-mini -T pyrit -r "target behavior"

# Scan Garak
python scripts/run_cloud_attack.py -t gpt-4o-mini -T garak --probes dan encoding
```

## Intel NPU (Core Ultra)

### Setup
```bash
# Vérifier status
python -m modules.hardware.npu_runtime --status

# Installer IPEX-LLM portable
python -m modules.hardware.npu_runtime --setup
```

### Modèles NPU supportés
- `Llama-3.2-3B-Instruct`
- `DeepSeek-R1-Distill-Qwen-1.5B`
- `DeepSeek-R1-Distill-Qwen-7B`

**Driver requis**: 32.0.100.3104+

## Structure

```
local-ai-agents/
├── modules/
│   ├── redteam/          # Attaques cloud (TAP, PyRIT, Garak)
│   ├── hardware/         # NPU Intel
│   ├── jailbreak/        # Techniques locales
│   └── agents/           # Agents LLM
├── scripts/              # CLI
├── configs/              # Configuration
└── tests/                # Résultats
```

## Documentation

- [CLAUDE.md](CLAUDE.md) - Instructions détaillées
- [ROADMAP.md](ROADMAP.md) - Vision R&D
- [docs/RESEARCH_2026.md](docs/RESEARCH_2026.md) - État de l'art

## Ressources

- [OpenAI GPT-5.2 Docs](https://platform.openai.com/docs/models/gpt-5.2)
- [Gemini 3 Dev Guide](https://ai.google.dev/gemini-api/docs/gemini-3)
- [IPEX-LLM NPU](https://github.com/intel/ipex-llm)

## Éthique

Recherche en sécurité autorisée uniquement. Responsible disclosure si vulnérabilité critique.
