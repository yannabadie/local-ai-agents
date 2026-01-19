# État de l'Art - Recherche Janvier 2026

> **Dernière mise à jour**: 19 janvier 2026
> **Objectif**: Éviter de répéter les recherches web

---

## 1. Modèles Cloud - État Actuel

### OpenAI (Janvier 2026)

**Source**: [OpenAI Models](https://platform.openai.com/docs/models/)

#### GPT-5.2 Series (Released December 2025)
- **gpt-5.2**: Flagship model, 90% ARC-AGI-1, 100% AIME 2025, 400K context
- **gpt-5.2-pro**: Maximum capability with xhigh reasoning effort
- **gpt-5.2-codex**: Optimized for agentic coding tasks

#### Pricing (per 1M tokens)
| Model | Input | Output | Cached Input |
|-------|-------|--------|--------------|
| gpt-5.2 | $1.75 | $14.00 | $0.175 (90% off) |
| gpt-5.2-pro | $10.00 | $40.00 | - |
| gpt-4o | $2.50 | $10.00 | - |
| gpt-4o-mini | $0.15 | $0.60 | - |
| o3-mini | $1.10 | $4.40 | - |

#### API Features
- Reasoning effort: none, low, medium, high, xhigh
- Responses API + Chat Completions API
- 400K context window
- gpt-5.1 available until March 2026 (legacy)

---

### Google Gemini (Janvier 2026)

**Source**: [Gemini 3 Dev Guide](https://ai.google.dev/gemini-api/docs/gemini-3)

#### Gemini 3 Series (Announced January 12, 2026)
- **gemini-3-pro-preview**: Reasoning-first, 1M context, adaptive thinking
- **gemini-3-flash-preview**: Pro-level intelligence at Flash speed, **FREE TIER**

#### Pricing (per 1M tokens)
| Model | Input | Output | Notes |
|-------|-------|--------|-------|
| gemini-3-pro-preview | $1.25 | $5.00 | Paid only |
| gemini-3-flash-preview | $0.075 | $0.30 | **Free tier available** |

#### New API Features
- `thinking_level` parameter for reasoning depth
- Thought Signatures for chain-of-reasoning persistence
- Google Gen AI SDK: `from google import genai`

#### Benchmarks
- GPQA Diamond: 90.4%
- Humanity's Last Exam: 33.7% (without tools)
- 1M token context, 64K output

---

### Anthropic Claude

**Note**: Pas de changements majeurs documentés pour janvier 2026.

| Model | Input | Output |
|-------|-------|--------|
| claude-3-opus | $15.00 | $75.00 |
| claude-3.5-sonnet | $3.00 | $15.00 |
| claude-3-haiku | $0.25 | $1.25 |

---

## 2. Intel NPU - État Actuel

**Source**: [IPEX-LLM GitHub](https://github.com/intel/ipex-llm)

### IPEX-LLM Versions
- **v2.2.0** (Stable): Ollama + llama.cpp portable zip
- **v2.3.0-nightly**: AMX, TP support, multimodal (mtmd-cli)

### NPU Support Status
- **Windows only** pour NPU
- Driver requis: **32.0.100.3104**
- Limitations: max 960 input tokens, 1024 total sequence

### Processeurs Supportés
| Series | Code Name | Config Requise |
|--------|-----------|----------------|
| 1xxH/U/HL/UL | Meteor Lake | `IPEX_LLM_NPU_MTL=1` |
| 2xxV | Lunar Lake | Aucune |
| 2xxK/2xxH | Arrow Lake | `IPEX_LLM_NPU_ARL=1` |

### Modèles NPU Vérifiés
**Seuls ces modèles sont officiellement supportés sur NPU:**
1. `meta-llama/Llama-3.2-3B-Instruct`
2. `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`
3. `deepseek-ai/DeepSeek-R1-Distill-Qwen-7B`

### Performance Notes
- iGPU souvent plus rapide que NPU pour LLM inference
- NPU optimal pour: static shapes, vision models, energy efficiency
- INT4 peut être plus lent sur NPU que CPU

---

## 3. Frameworks Red-Team

### PyRIT (Microsoft)
- **Repo**: https://github.com/Azure/PyRIT
- **Features**: Bulk prompt generation, multi-turn, orchestration
- **Install**: `pip install pyrit`

### Garak (NVIDIA)
- **Repo**: https://github.com/NVIDIA/garak
- **Features**: Vulnerability scanner, probes, detectors
- **Install**: `pip install garak`

### TAP (Tree of Attacks with Pruning)
- **Paper**: NeurIPS 2024
- **Success Rate**: >80% on GPT-4 (historique)
- **Queries**: ~28 average
- **Principe**: Iterative generation with ineffective branch pruning

---

## 4. Techniques d'Attaque Efficaces (2026)

| Technique | Success Rate | Target | Source |
|-----------|-------------|--------|--------|
| TAP | >80% | GPT-4 (historique) | NeurIPS 2024 |
| PAIR | Competitive | GPT-4, Claude | Black-box |
| AutoDAN-Turbo | High | Reasoning models | ICLR 2025 |
| Multi-modal injection | Variable | GPT-4V, Gemini | Visual bypass |

---

## 5. Liens de Référence

### Documentation Officielle
- [OpenAI API Models](https://platform.openai.com/docs/models/)
- [OpenAI Changelog](https://platform.openai.com/docs/changelog)
- [Gemini API Models](https://ai.google.dev/gemini-api/docs/models)
- [Gemini 3 Guide](https://ai.google.dev/gemini-api/docs/gemini-3)
- [IPEX-LLM NPU Quickstart](https://github.com/intel/ipex-llm/blob/main/docs/mddocs/Quickstart/npu_quickstart.md)
- [IPEX-LLM Releases](https://github.com/intel/ipex-llm/releases)

### Papers & Research
- TAP: "Tree of Attacks" - NeurIPS 2024
- AutoDAN-Turbo - ICLR 2025
- NITRO: LLM Inference on Intel NPU - arXiv:2412.11053

---

## 6. À Vérifier Prochainement

- [ ] Claude 4 announcement (rumored Q1 2026)
- [ ] GPT-5.3 / GPT-6 roadmap
- [ ] Gemini 3 GA (currently preview)
- [ ] IPEX-LLM v2.3.0 stable release
- [ ] New NPU models support

---

*Ce document est mis à jour manuellement après chaque session de recherche.*
