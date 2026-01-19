# Hardware Profile

## Specifications

| Component | Details |
|-----------|---------|
| **CPU** | Intel Core Ultra 5 135U |
| **Architecture** | Meteor Lake |
| **Cores/Threads** | 12 cores / 14 threads |
| **Base Clock** | 1.6 GHz |
| **NPU** | Intel AI Boost (intégré) |
| **RAM** | 16 GB |
| **GPU** | Intel Graphics (intégré) |
| **VRAM** | ~2 GB (partagée) |
| **Stockage** | 475 GB SSD (150 GB libre) |
| **OS** | Windows 11 Professionnel |

## Capacités et limites

### Points forts
- **NPU Intel AI Boost** : Accélérateur IA dédié, potentiellement exploitable via IPEX-LLM/OpenVINO
- **16 GB RAM** : Suffisant pour modèles 7-8B quantifiés Q4
- **12 cores** : Bon parallélisme pour inference CPU
- **AVX2/AVX-512** : Instructions vectorielles supportées par llama.cpp

### Limites
- **Pas de GPU dédié** : Inference lente (30s-2min par réponse)
- **2 GB VRAM partagée** : Insuffisant pour offload GPU significatif
- **150 GB libre** : Limite le nombre de modèles stockables

## Modèles compatibles

| Taille modèle | Quantization | RAM requise | Compatible |
|---------------|--------------|-------------|------------|
| 3B | Q4_K_M | ~3 GB | Oui (fluide) |
| 7B | Q4_K_M | ~5 GB | Oui |
| 7B | Q8_0 | ~8 GB | Oui |
| 8B | Q4_K_M | ~5 GB | Oui |
| 13B | Q4_K_M | ~8 GB | Limite |
| 13B | Q8_0 | ~14 GB | Risqué |
| 20B+ | Any | >16 GB | Non |

## Optimisations possibles

### Court terme
- Utiliser quantization Q4_K_M (meilleur ratio qualité/vitesse)
- Configurer `num_threads` optimal dans Ollama (12)
- Activer mmap pour réduire empreinte mémoire

### Moyen terme
- Explorer IPEX-LLM pour accélération Intel
- Tester OpenVINO backend pour NPU
- Évaluer llama.cpp avec SYCL backend

### Ressources
- [IPEX-LLM](https://github.com/intel/ipex-llm)
- [Intel NPU Acceleration](https://github.com/intel/intel-npu-acceleration-library)
- [llama.cpp NPU Discussion](https://github.com/ggml-org/llama.cpp/discussions/15883)
