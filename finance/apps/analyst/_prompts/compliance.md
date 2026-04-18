Review the following closed trades and evaluate compliance with the swing trading playbook.

For each trade, assess:
1. **Entry compliance**: Was the entry consistent with the stated strategy and setup type rules?
2. **Management compliance**: Was actual management consistent with intended management?
3. **Exit compliance**: Did the exit follow the correct exit signal, or was it premature/late?
4. **Optimal trade**: What would the playbook-perfect trade have looked like? (entry, structure, stop, exit)
5. **Intuition vs rules**: Did any deviations from the rules actually produce better results? Flag these as strategy refinement opportunities.

# Market Conditions at Time of Trades

{market_context}

# Trades to Review

{trades}

# Required JSON Output

```json
[
  {
    "trade_id": 123,
    "symbol": "TICKER",
    "score": 4,
    "analysis": "Full markdown analysis with sections:\n\n## Entry Compliance\n...\n\n## Management Compliance\n...\n\n## Exit Compliance\n...\n\n## Optimal Trade\n...\n\n## Strategy Refinements\n..."
  }
]
```

The `analysis` field must be markdown text with the sections above. The `score` is 1-5:
- 5: Perfect playbook execution
- 4: Minor deviations, no material impact
- 3: Notable deviations but reasonable judgment
- 2: Significant rule violations
- 1: Trade contradicted the playbook

After individual reviews, add an aggregate summary:

```json
{
  "aggregate": {
    "avg_score": 3.5,
    "patterns": ["Recurring patterns across trades"],
    "top_improvement": "The single most impactful improvement suggestion",
    "refinements": ["Where intuition consistently outperformed rules"]
  }
}
```
