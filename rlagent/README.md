# Group E tournament agent (Section 2.4)

`agentE.py` provides `policy(t, B_self, B_opp, S_self, S_opp, g, R) -> (delta, x)`.
It needs only NumPy, loads its weights from `agentE_theta.npy` in this folder,
does not learn during play and uses only the current state.
