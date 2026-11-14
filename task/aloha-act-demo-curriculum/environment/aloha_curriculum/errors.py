"""Error hierarchy shared by the runtime.

``CandidateInvalid`` marks a violation of the task contract by the agent's candidate
(policy.json / scores.csv). The Judge reports it as a candidate error, not a score.
Everything else that goes wrong is an infrastructure error.
"""


class CandidateInvalid(ValueError):
    """The candidate violates the task contract; it is not scoreable."""
