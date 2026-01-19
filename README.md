# Local AI Agents - LLM Security Research Framework

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Research](https://img.shields.io/badge/Research-Security-red.svg)](docs/research/)

> **A comprehensive framework for testing LLM security guardrails with reproducible POCs**

This project provides tools and documented research for testing the security of Large Language Models, both cloud-based (GPT, Claude, Gemini) and locally deployed (Ollama). Unlike theoretical collections, this repository contains **actual working exploits with reproducible test results**.

## Key Results (January 2026)

| Technique | Target | Success Rate | Avg. Queries | Evidence |
|-----------|--------|--------------|--------------|----------|
| **Policy Puppetry** | GPT-4.1-mini | Variable* | 1 | [tests/poc/jailbreak-2026/](tests/poc/jailbreak-2026/) |
| **Policy Puppetry** | Gemini-3-flash | Variable* | 1 | [tests/poc/jailbreak-2026/](tests/poc/jailbreak-2026/) |
| **Involuntary Jailbreak** | GPT-4.1-mini | High | 1 | [docs/research/](docs/research/) |
| **Direct Query** | deephat (local) | 100% | 1 | Local evidence |
| **Direct Query** | elbaz-olmo (local) | 100% | 1 | Local evidence |

*60 reproducible tests across 5 prompt variations (default, XML, JSON, roleplay, leetspeak)

## Features

### Reproducible Testing Framework
```python
from tests.poc.jailbreak_2026.reproducible_tests import ReproducibleTester

tester = ReproducibleTester("My_Experiment")
results = tester.run_cloud_test(
    technique_name="policy_puppetry",
    prompt_generator=policy_puppetry_prompt,
    target_behaviors=["SQL injection techniques"],
    models=[("openai", "gpt-4.1-mini"), ("google", "gemini-3-flash-preview")],
    num_runs=3,  # Multiple runs for statistical significance
    variations=["default", "xml", "json"]
)
# Results saved: local (full) + public (anonymized)
```

### SOTA Attack Techniques (2026)

| Technique | Paper | Success Rate | Description |
|-----------|-------|--------------|-------------|
| **JBFuzz** | [arXiv:2503.08990](https://arxiv.org/abs/2503.08990) | 99% in 60s | Fuzzing-based, ~7 queries avg |
| **Policy Puppetry** | HiddenLayer 2025 | Universal | XML/JSON policy injection |
| **Involuntary Jailbreak** | Wei et al. 2024 | High | Self-prompting attack |
| **TAP** | NeurIPS 2024 | >80% GPT-4 | Tree of Attacks with Pruning |
| **AutoDAN-Turbo** | ICLR 2025 | High | Genetic algorithms |

### Metacognition Module
Self-verification framework for research integrity:
```python
from modules.core.metacognition import MetacognitionEngine

engine = MetacognitionEngine("research_session")
engine.record_thought(
    content="Policy Puppetry works via XML config interpretation",
    confidence=0.85,
    reasoning="Confirmed by 60 tests",
    evidence=["results_local/policy_puppetry.json"]
)
```

### Multi-Provider Cloud Testing
```bash
# Test all providers
python scripts/run_cloud_attack.py --test-connection

# Attack specific model
python scripts/run_cloud_attack.py -t gpt-5-nano -T tap -r "target behavior"

# List available models
python scripts/run_cloud_attack.py --list-models
```

## Quick Start

```bash
# Clone
git clone https://github.com/yannabadie/local-ai-agents.git
cd local-ai-agents

# Install dependencies
pip install -r requirements.txt

# Configure API keys (.env)
cp .env.example .env
# Edit .env with your keys:
# OPENAI_API_KEY=sk-...
# GOOGLE_API_KEY=AIza...

# Run reproducible tests
python -m tests.poc.jailbreak_2026.reproducible_tests
```

## Supported Models

### Cloud APIs (January 2026)

| Provider | Model | Cost (1M tokens) | Notes |
|----------|-------|------------------|-------|
| OpenAI | `gpt-5-nano` | $0.10 / $0.40 | Fastest, cheapest |
| OpenAI | `gpt-5-mini` | $0.30 / $1.20 | Good reasoning |
| OpenAI | `gpt-4.1-mini` | $0.40 / $1.60 | Legacy, tested |
| Google | `gemini-3-flash-preview` | $0.075 / $0.30 | **Free tier** |
| Anthropic | `claude-haiku-4-5` | $0.25 / $1.25 | Fast Claude |

### Local Models (Ollama)

| Model | Command | Type |
|-------|---------|------|
| DeepHat-V1-7B | `ollama run deephat` | Cybersecurity |
| Elbaz-OLMo-3-7B | `ollama run elbaz-olmo` | Uncensored |
| DeepSeek-R1-Distill-8B | `ollama run deepseek-r1` | Reasoning |

## Project Structure

```
local-ai-agents/
├── modules/
│   ├── redteam/           # Cloud attack implementations
│   │   ├── cloud_client.py    # Unified API client
│   │   └── tap/               # TAP attack implementation
│   ├── core/              # Core utilities
│   │   └── metacognition.py   # Self-verification framework
│   ├── agents/            # LLM agents
│   │   └── environment_agent.py  # Local interaction
│   └── jailbreak/         # Jailbreak techniques
├── tests/
│   └── poc/
│       └── jailbreak-2026/    # Reproducible POCs
│           ├── reproducible_tests.py
│           ├── policy_puppetry_test.py
│           └── results_public/  # Anonymized results
├── docs/
│   └── research/
│       └── JAILBREAK_TECHNIQUES_2026.md
├── scripts/               # CLI tools
└── configs/               # Configuration files
```

## Research Documentation

- [JAILBREAK_TECHNIQUES_2026.md](docs/research/JAILBREAK_TECHNIQUES_2026.md) - Comprehensive technique documentation
- [CLAUDE.md](CLAUDE.md) - Detailed project instructions
- [ROADMAP.md](ROADMAP.md) - R&D vision and priorities

## Comparison with Other Tools

| Feature | This Repo | Garak | PyRIT | JailbreakBench |
|---------|-----------|-------|-------|----------------|
| Reproducible POCs | Yes | Partial | Partial | Yes |
| Cloud + Local | Yes | Yes | Yes | Cloud only |
| 2026 Techniques | Yes | Partial | Partial | Yes |
| Self-verification | Yes | No | No | No |
| Anonymized results | Yes | No | No | No |
| MIT License | Yes | Apache 2.0 | MIT | MIT |

## Ethical Guidelines

This research is conducted for **defensive purposes**:
- All tests on personal API accounts
- Findings documented for improving LLM safety
- Responsible disclosure for critical vulnerabilities
- No malicious use or distribution of exploits

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add reproducible tests with anonymized results
4. Submit a pull request

## License

MIT License - see [LICENSE](LICENSE)

## References

- [JBFuzz Paper (arXiv:2503.08990)](https://arxiv.org/abs/2503.08990)
- [NVIDIA Garak](https://github.com/NVIDIA/garak)
- [Microsoft PyRIT](https://github.com/Azure/PyRIT)
- [JailbreakBench](https://github.com/JailbreakBench/jailbreakbench)
- [Awesome-Jailbreak-on-LLMs](https://github.com/yueliu1999/Awesome-Jailbreak-on-LLMs)
- [OWASP LLM Top 10 2025](https://owasp.org/www-project-top-10-for-large-language-model-applications/)

---

**Disclaimer**: This project is for authorized security research only. Users are responsible for compliance with applicable laws and terms of service.
