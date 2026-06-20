# BettaFish-LG

> A LangGraph refactor of [BettaFish](https://github.com/666ghj/BettaFish) — production-grade orchestration for multi-agent sentiment analysis.

**🚧 v0.1 in development**

## Why this fork

The original BettaFish demonstrates excellent zero-framework multi-agent collaboration. This refactor addresses three production limitations:

1. **Orchestration**: Multi-subprocess Streamlit architecture → single-process LangGraph StateGraph
2. **Robustness**: No counter-evidence mechanism → adversarial Devil's Advocate agent
3. **Quality**: No evaluation harness → RAGAS-based eval

## Quick start

```bash
git clone https://github.com/duukkid-arch/BettaFish-LG.git
cd BettaFish-LG
pip install -e .
cp .env.example .env  # fill in DashScope and Tavily keys
python scripts/run.py "DeepSeek V3"
```

## Roadmap

- [x] Phase 1: LangGraph scaffold and supervisor
- [ ] Phase 2: Real agent subgraphs
- [ ] Phase 3: Devil's Advocate + SQLite memory
- [ ] Phase 4: RAGAS evaluation suite

## Citing

Derived from [666ghj/BettaFish](https://github.com/666ghj/BettaFish). See [NOTICE.md](./NOTICE.md).