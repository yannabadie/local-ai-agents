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
python scripts/run_cloud_attack.py -t gpt-5-nano -T tap -r "explain lockpicking"
```

## Modèles Cloud (Janvier 2026 - VERIFIED)

### OpenAI
| Modèle | Prix (per 1M tokens) | Use Case |
|--------|---------------------|----------|
| `gpt-5.2` | $1.75 / $14 | Flagship - 93.2% GPQA Diamond, 100% AIME |
| `gpt-5.2-pro` | $10 / $40 | xhigh reasoning effort |
| `gpt-5.2-codex` | $1.75 / $14 | Agentic coding, SWE-Bench Pro leader |
| `gpt-5-mini` | $0.30 / $1.20 | Powerful reasoning at lower cost |
| `gpt-5-nano` | $0.10 / $0.40 | **FASTEST, most affordable** |
| `gpt-4.1` | $2.00 / $8.00 | 1M context, fine-tuning available |

### Google Gemini 3 (January 2026)
| Modèle | Prix (per 1M tokens) | Use Case |
|--------|---------------------|----------|
| `gemini-3-pro-preview` | $2.00 / $12 | Reasoning-first, 1M context |
| `gemini-3-flash-preview` | $0.075 / $0.30 | **FREE TIER**, 90.4% GPQA Diamond |

### Anthropic Claude 4.5 (November 2025)
| Modèle | Prix (per 1M tokens) | Use Case |
|--------|---------------------|----------|
| `claude-opus-4-5-20251101` | $5 / $25 | Flagship - 99.78% harmless |
| `claude-sonnet-4-5` | $3 / $15 | Balanced performance |
| `claude-haiku-4-5` | $0.25 / $1.25 | **Fastest Claude** |

> **Note**: Claude Opus 4 and 4.1 have been REMOVED from the API.

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
    --target gpt-5-nano \
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
python scripts/run_cloud_attack.py -t gpt-5-nano -T pyrit -r "target behavior"

# Scan Garak
python scripts/run_cloud_attack.py -t gpt-5-nano -T garak --probes dan encoding
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
- [docs/METACOGNITION.md](docs/METACOGNITION.md) - Framework de recherche

## Ressources (Verified January 2026)

- [OpenAI Models](https://platform.openai.com/docs/models/)
- [Gemini 3 Dev Guide](https://ai.google.dev/gemini-api/docs/gemini-3)
- [Anthropic Claude](https://www.anthropic.com/claude)
- [IPEX-LLM NPU](https://github.com/intel/ipex-llm)

## Éthique

Recherche en sécurité autorisée uniquement. Responsible disclosure si vulnérabilité critique.
