# Local AI Agents - Cybersecurity & Uncensored Models Testing

Testing capabilities of on-premise uncensored and cybersecurity-oriented LLMs with environment interaction.

## Objectives

- Test uncensored/abliterated models capabilities
- Test cybersecurity-focused models (red/blue team, threat analysis)
- Enable efficient environment interaction (shell, files, APIs)
- Compare model performance on security-related tasks

## Installed Models

| Model | Size | Specialization |
|-------|------|----------------|
| `deephat` | 4.7 GB | Cybersecurity, DevOps, threat analysis |
| `elbaz-olmo` | 4.5 GB | General purpose (uncensored) |
| `deepseek-r1` | 4.9 GB | Reasoning, math, code |

## Stack

- **Runtime**: Ollama
- **Agent Framework**: Open Interpreter
- **OS**: Windows 11

## Usage

```bash
# Start a model
ollama run deephat

# With Open Interpreter
interpreter --api_base "http://localhost:11434" --model "ollama/deephat"
```

## Project Structure

```
local-ai-agents/
├── README.md
├── configs/           # Model configurations
├── scripts/           # Automation scripts
├── tests/             # Test scenarios
└── docs/              # Documentation
```

## License

MIT
