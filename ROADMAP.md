# ROADMAP R&D - Local AI Agents

> Recherche en sécurité IA : modèles uncensored, contournement de garde-fous, interaction environnement

## Vision

Créer un laboratoire local de recherche en sécurité IA capable de :
- Tester les limites des modèles LLM sans restrictions cloud
- Développer et documenter des techniques de contournement pour améliorer les garde-fous
- Permettre l'interaction autonome modèle-environnement
- Contribuer à la recherche en sécurité IA

---

## Axes de recherche parallèles

### AXE A : Catalogue de modèles uncensored

**Objectif** : Constituer une base de modèles uncensored/abliterated performants et documentés

#### A.1 Modèles prioritaires à évaluer

| Modèle | Source | Taille | Priorité | Statut |
|--------|--------|--------|----------|--------|
| DeepHat-V1-7B | Kindo.ai | 4.7 GB | Installé | ✅ |
| Elbaz-OLMo-3-7B-abliterated | Ex0bit | 4.5 GB | Installé | ✅ |
| Dolphin-3.0-Llama-3.1-8B | Cognitive Computations | ~5 GB | Haute | ⬜ |
| L3.2-Rogue-Creative-Uncensored-7B | DavidAU | ~5 GB | Haute | ⬜ |
| Nous-Hermes-3-Llama-3.2-8B | NousResearch | ~5 GB | Moyenne | ⬜ |
| Qwen2.5-7B-Abliterated | Community | ~5 GB | Moyenne | ⬜ |
| Mistral-7B-Dolphin | Cognitive Computations | ~4.5 GB | Moyenne | ⬜ |

#### A.2 Veille continue

**Hypothèse** : De nouveaux modèles abliterated apparaissent régulièrement sur HuggingFace

**Expériences** :
- [ ] Créer un script de veille HuggingFace (tags: abliterated, uncensored, GGUF)
- [ ] Monitorer les créateurs prolifiques : DavidAU, mlabonne, NousResearch, Cognitive Computations
- [ ] Documenter chaque modèle testé dans `docs/models/`

**Métriques** :
- Taux de compliance aux requêtes "unsafe" (baseline: modèles standard ~18%, cible: >70%)
- Qualité des réponses (cohérence, utilité)
- Performance inference (tokens/sec sur notre hardware)

---

### AXE B : Techniques de contournement (Red Teaming)

**Objectif** : Documenter et tester les techniques de jailbreak/prompt injection pour améliorer les défenses

#### B.1 Taxonomie des attaques

| Catégorie | Technique | Efficacité rapportée | À tester |
|-----------|-----------|---------------------|----------|
| **Roleplay** | DAN (Do Anything Now) | ~89% ASR | ⬜ |
| **Roleplay** | Grandma exploit | Haute | ⬜ |
| **Roleplay** | Persona injection | Haute | ⬜ |
| **Logic traps** | Dilemmes moraux | ~81% ASR | ⬜ |
| **Logic traps** | Conditional structures | Moyenne | ⬜ |
| **Encoding** | Base64 prompts | ~76% ASR | ⬜ |
| **Encoding** | Zero-width characters | Haute | ⬜ |
| **Encoding** | Homoglyphs | Moyenne | ⬜ |
| **Multi-turn** | Context manipulation | Haute | ⬜ |
| **Multi-turn** | Instruction splitting | Moyenne | ⬜ |

#### B.2 Expériences structurées

**Protocole de test** :
1. Définir un jeu de prompts "unsafe" standardisé (AdvBench, HarmBench)
2. Tester chaque technique sur chaque modèle (matrice complète)
3. Mesurer : taux de succès, qualité de réponse, détectabilité
4. Documenter dans `tests/jailbreak/`

**Hypothèses à valider** :
- [ ] H1: Les modèles abliterated sont-ils immunisés aux techniques de jailbreak ? (probable: oui)
- [ ] H2: Les techniques multi-turn sont-elles plus efficaces que single-turn ?
- [ ] H3: La combinaison de techniques augmente-t-elle le taux de succès ?

#### B.3 Recherche avancée

- [ ] Étudier la technique d'abliteration (paper Arditi et al.)
- [ ] Tenter d'abliterate un modèle nous-mêmes (Qwen3, Phi-4)
- [ ] Explorer les "refusal directions" dans l'espace latent
- [ ] Tester la résistance des nouveaux modèles (Gemma 3, Qwen 3, Phi 4)

**Outils** :
- [llm-abliteration](https://github.com/NousResearch/llm-abliteration)
- [DeepTeam](https://github.com/confident-ai/deepteam) - Red teaming framework
- [Promptfoo](https://www.promptfoo.dev/) - Testing framework

---

### AXE C : Interaction environnement (Agents)

**Objectif** : Permettre aux modèles d'exécuter des actions sur le système local

#### C.1 Frameworks à évaluer

| Framework | Type | Ollama compatible | Priorité |
|-----------|------|-------------------|----------|
| Open Interpreter | Code execution | ✅ | Haute |
| Goose | Dev agent | ✅ | Haute |
| Observer AI | Screen + system | ✅ | Moyenne |
| LangGraph + Ollama | Custom agents | ✅ | Moyenne |
| AIlice | General purpose | ✅ | Basse |

#### C.2 Expériences

**Phase 1 : Capacités de base**
- [ ] Installer Open Interpreter avec Ollama
- [ ] Tester : lecture/écriture fichiers, exécution shell, navigation web
- [ ] Mesurer : taux de réussite, dangerosité des actions

**Phase 2 : Agents autonomes**
- [ ] Configurer des tâches complexes multi-étapes
- [ ] Tester la persistance et la mémoire
- [ ] Évaluer la capacité de planification

**Phase 3 : Agents offensifs (environnement isolé)**
- [ ] Créer une VM sandbox pour tests
- [ ] Tester des scénarios red team automatisés
- [ ] Documenter les capacités et limites

#### C.3 Sécurité MCP (Model Context Protocol)

**Contexte** : MCP est le protocole émergent pour l'interaction LLM-outils, avec des vulnérabilités connues

**Recherche** :
- [ ] Étudier les CVE MCP récentes (CVE-2025-6514, etc.)
- [ ] Tester les attaques : tool poisoning, prompt injection via MCP
- [ ] Documenter les mitigations

**Sources** :
- [MCP Security Best Practices](https://modelcontextprotocol.io/specification/draft/basic/security_best_practices)
- [Unit42 MCP Attack Vectors](https://unit42.paloaltonetworks.com/model-context-protocol-attack-vectors/)

---

### AXE D : Optimisation hardware

**Objectif** : Maximiser les performances sur le hardware disponible (Intel Core Ultra 5 + NPU)

#### D.1 Optimisations CPU

- [ ] Benchmark baseline avec Ollama default
- [ ] Tester différentes valeurs `num_threads` (8, 12, 14)
- [ ] Comparer quantizations : Q4_K_M vs Q4_K_S vs Q5_K_M
- [ ] Évaluer impact du context size sur la vitesse

#### D.2 Exploitation du NPU Intel

**Hypothèse** : Le NPU peut accélérer l'inference jusqu'à 2-3x

**Expériences** :
- [ ] Installer IPEX-LLM
- [ ] Tester llama.cpp avec backend NPU (via portable zip Intel)
- [ ] Comparer : CPU seul vs CPU+NPU
- [ ] Documenter les limitations (max 1024 tokens séquence)

**Ressources** :
- [IPEX-LLM GitHub](https://github.com/intel/ipex-llm)
- [llama.cpp NPU Quickstart](https://github.com/intel/ipex-llm/blob/main/docs/mddocs/Quickstart/llama_cpp_npu_portable_zip_quickstart.md)

#### D.3 Modèles optimaux pour ce hardware

**Critères** : <6 GB RAM, >1 token/sec, qualité acceptable

| Catégorie | Meilleur candidat | Backup |
|-----------|-------------------|--------|
| Uncensored général | Dolphin-3.0-8B Q4 | Elbaz-OLMo Q4 |
| Cybersécurité | DeepHat-7B Q4 | - |
| Raisonnement | DeepSeek-R1-8B Q4 | Qwen3-8B Q4 |
| Créatif | L3.2-Rogue-7B Q4 | - |

---

### AXE E : Documentation et contribution

**Objectif** : Documenter les découvertes et contribuer à la communauté

#### E.1 Structure documentation

```
docs/
├── models/           # Fiches par modèle testé
├── techniques/       # Techniques de jailbreak documentées
├── benchmarks/       # Résultats de tests
├── tutorials/        # Guides pratiques
└── research/         # Notes de recherche
```

#### E.2 Livrables potentiels

- [ ] Matrice comparative modèles uncensored (public)
- [ ] Guide d'abliteration pour débutants
- [ ] Benchmark jailbreak techniques 2026
- [ ] Article : "Red teaming local LLMs on consumer hardware"

---

## Matrice de priorisation

| Axe | Impact | Effort | Priorité |
|-----|--------|--------|----------|
| A - Catalogue modèles | Haut | Faible | 🔴 P0 |
| C - Agents | Haut | Moyen | 🔴 P0 |
| B - Red teaming | Haut | Moyen | 🟠 P1 |
| D - Optimisation hardware | Moyen | Moyen | 🟠 P1 |
| E - Documentation | Moyen | Faible | 🟡 P2 |

---

## Quick wins (démarrage immédiat)

1. **Installer Open Interpreter** et tester avec DeepHat
2. **Télécharger Dolphin-3.0-8B** (modèle uncensored de référence)
3. **Créer premier jeu de test** jailbreak (10 prompts)
4. **Benchmark baseline** tokens/sec pour chaque modèle installé

---

## Ressources clés

### Papers
- [Refusal in LLMs is mediated by a single direction](https://www.lesswrong.com/posts/jGuXSZgv6qfdhMCuJ/refusal-in-llms-is-mediated-by-a-single-direction) - Arditi et al.
- [Uncensored AI in the Wild](https://www.mdpi.com/1999-5903/17/10/477) - Tracking abliterated models
- [Bypassing LLM Guardrails](https://arxiv.org/abs/2504.11168) - Evasion techniques

### Outils
- [Abliteration Tutorial](https://huggingface.co/blog/mlabonne/abliteration) - mlabonne
- [DeepTeam](https://github.com/confident-ai/deepteam) - Red teaming framework
- [IPEX-LLM](https://github.com/intel/ipex-llm) - Intel acceleration

### Communautés
- [LocalLLaMA Reddit](https://reddit.com/r/LocalLLaMA)
- [HuggingFace Abliterated Models](https://huggingface.co/models?search=abliterated)
- DavidAU sur HuggingFace (prolifique créateur de modèles abliterated)

---

## Changelog

| Date | Modification |
|------|--------------|
| 2026-01-19 | Création initiale |
