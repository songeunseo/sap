# DSA graph search adaptation on LLaDA @65%

Not an exact reproduction of DSA paper search: the public repository lacks its
PPL-guided graph evolution controller. Public graph operators/mapping retained;
our bounded evolution controller uses equal-state mean masked gold CE on frozen
80 DLM states. No GSM8K fitness. Seed0/population8/generations4/elites2/top4parents,
stage crossover and mutation probability .5, fixed Lambda .08. Physically unsafe
matrix operations on full pooled vectors excluded; numerical failures rejected.
Same224 targets/Wanda ranking/exact65% budget. Freeze winner before heldout40
diagnostic and historical mini100. No automatic full. Original7/100 preserved.

GPU1 tmux dsa_search65. Status: python3 experiments/dlm_dsa_search65/status.py
