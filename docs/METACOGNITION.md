# Framework de Métacognition - OBLIGATOIRE

> **RÈGLE ABSOLUE**: Mes données d'entraînement sont OBSOLÈTES par défaut.
> Je DOIS faire des recherches web AVANT toute décision technique.

---

## Checklist AVANT toute action technique

### 1. RECHERCHER D'ABORD
```
□ Recherche web sur l'état actuel de la technologie
□ Vérifier les versions/modèles disponibles AUJOURD'HUI
□ Consulter la documentation officielle actuelle
□ Vérifier les dates des sources (< 1 mois = fiable)
```

### 2. VALIDER LES INFORMATIONS
```
□ Croiser minimum 2 sources
□ Privilégier: docs officielles > blogs techniques > mes connaissances
□ Si doute → rechercher encore
```

### 3. DOCUMENTER
```
□ Sauvegarder les findings dans docs/RESEARCH_2026.md
□ Inclure les URLs sources
□ Dater chaque mise à jour
```

---

## État de l'Art - Janvier 2026

### OpenAI API Models (Vérifié 19/01/2026)

**Source**: [OpenAI Models](https://platform.openai.com/docs/models/)

| Génération | Modèles | Status |
|------------|---------|--------|
| **GPT-5.2** | gpt-5.2, gpt-5.2-pro, gpt-5.2-codex | Flagship |
| **GPT-5.1** | gpt-5.1, gpt-5.1-codex | Available |
| **GPT-5** | gpt-5, gpt-5-mini, gpt-5-nano | Available |
| **GPT-4.1** | gpt-4.1, gpt-4.1-mini, gpt-4.1-nano | **Remplace GPT-4o** |
| **o-series** | o3-mini | Reasoning |

⚠️ **GPT-4o est OBSOLÈTE** - Utiliser GPT-4.1 ou GPT-5.x

### Google Gemini API (Vérifié 19/01/2026)

**Source**: [Gemini Models](https://ai.google.dev/gemini-api/docs/models)

| Génération | Modèles | Status |
|------------|---------|--------|
| **Gemini 3** | gemini-3-pro-preview, gemini-3-flash-preview | Preview |
| **Gemini 2.5** | gemini-2.5-pro, gemini-2.5-flash, gemini-2.5-flash-lite | **Stable** |
| **Gemini 2.0** | gemini-2.0-flash | Legacy |

### Techniques Jailbreak 2026

**Sources**: [HackAIGC](https://www.hackaigc.com/blog/the-latest-trends-breakthroughs-of-llm-jailbreak-techniques-in-2026)

| Technique | Success Rate | Queries | Source |
|-----------|-------------|---------|--------|
| JBFuzz | **99%** | ~60s | arXiv 2503.08990 |
| Bad Likert Judge | +75% vs baseline | Multi-turn | Unit42 |
| Deceptive Delight | 65% | 3 turns | Unit42 |
| PAIR | Competitive | <20 | jailbreaking-llms.github.io |

### Outils Red-Team 2026

**Source**: [Mindgard](https://mindgard.ai/blog/best-tools-for-red-teaming)

| Outil | Maintainer | Spécialité |
|-------|------------|------------|
| PyRIT | Microsoft | Automation, enterprise |
| Promptfoo | Open Source | Dev-first, OWASP/NIST |
| Garak | NVIDIA | 100+ attack vectors |
| ART | IBM | Robustness testing |

---

## Anti-Patterns à Éviter

❌ Utiliser des modèles de ma mémoire sans vérifier
❌ Référencer GPT-4, GPT-4o, GPT-4-turbo (obsolètes)
❌ Assumer que les APIs n'ont pas changé
❌ Ignorer les deprecation notices
❌ Coder sans rechercher l'état de l'art

---

## Mise à jour de ce document

**Dernière recherche**: 19 janvier 2026
**Prochaine vérification obligatoire**: Avant chaque session majeure

```bash
# Commande pour forcer la recherche
# À exécuter mentalement avant toute décision technique
echo "STOP - Ai-je vérifié l'état actuel de cette technologie?"
```
