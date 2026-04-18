Evaluate the following scanner candidates against the swing trading playbook.

For each candidate, determine:
1. Which setup type (A/B/C/D) fits best, if any
2. Which profit mechanism (PM-01 through PM-05) applies
3. Whether Box 4 (Catalyst) can be assessed from available context
4. Recommended options structure based on the IVR decision tree
5. Specific entry, stop, and target levels

# Market Context

{market_context}

# Candidates

{candidates}

# Required JSON Output

```json
[
  {
    "symbol": "TICKER",
    "setup_type": "A | B | C | D | none",
    "profit_mechanism": "PM-01 | PM-02 | PM-03 | PM-04 | PM-05 | none",
    "thesis": "Why this trade fits the playbook",
    "catalyst_assessment": "Box 4 evaluation — what catalyst exists or is missing",
    "recommended_structure": "e.g. Long ATM call 45 DTE / Bull call spread",
    "entry": 0.00,
    "stop": 0.00,
    "target": 0.00,
    "risk_reward": "e.g. 3.2:1",
    "confidence": "high | medium | low",
    "reasoning": "Detailed explanation referencing specific playbook rules"
  }
]
```

Only include candidates that have at least a medium confidence of fitting a setup. Skip candidates that clearly don't fit any setup — mention them briefly at the end in a "skipped" list with one-line reasons.
