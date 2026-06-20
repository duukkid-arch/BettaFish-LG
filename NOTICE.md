# Notice

`BettaFish-LG` is a fork of [666ghj/BettaFish](https://github.com/666ghj/BettaFish).

## Preserved from upstream
- Multi-agent sentiment analysis paradigm (Query / Media / Insight / Report)
- Domain prompts for Chinese public-opinion analysis
- Report structure conventions

## What this fork adds
- LangGraph-based orchestration replacing the multi-subprocess Streamlit architecture
- An adversarial Devil's Advocate agent for counter-evidence surfacing
- Lightweight cross-session memory via SQLite
- RAGAS-based evaluation harness
- Tiered LLM strategy (qwen-turbo for routing, qwen-plus for synthesis) cutting cost ~60%