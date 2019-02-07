# Soft Actor-Critic
### Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor



**Two major challenges:**

- model-free methods have poor sample efficiency
- often brittle with respect to hyperparameters



**Other issues:**

- on-policy learning requires a sample to be collected for each step and it becomes difficult to collect many samples per step on a complex task
- off-policy learning + high-dimensional non-linear function approximation is a challenge for stability and convergence (further enhances in continuous state and action spaces)
- DDPG solves sample efficiency but is sensitive to hyper-parameters



**Solution objectives:**

- stable model-free RL algorithm for continuous state and action space
- augment maximum reward objective with and entropy maximisation term
- off-policy: enables reuse of previously collected data
- maximum entropy: to enable stability and exploration
- actor-critic architecture: separate policy and value network



**Maximum entropy reinforcement learning:**

- Standard RL maximizes expected sum of reward:
  $$
  \sum_{t} \mathbb{E} [r(s_t, a_t)]
  $$


- Maximum entropy objective which adds expected entropy of the policy over $\rho_\pi(s_t)$
  $$
  J(\pi) = \sum_{t=0}^{T} \mathbb{E}[r(s_t, a_t) + \alpha \mathcal{H}(\pi(.|s_t)) ]
  $$

  - policy is incentivized to explore widely, while not looking a unpromising regions
  - in cases where multiple actions looks optimal it will assign them equal probability
  - this can be extended to infinite horizon with a discount factor

- Soft actor-critic not solving for the Q-function directly

  - evaluate the Q-function for current policy
  - update the policy through an off-policy gradient update



Derivation of soft policy iteration:

- TODO: add proof for soft policy evaluation, improvement and both



**Soft Actor-Critic:**

- use function approximators for Q-function and policy

- alternate between optimizing both networks with SGD

- parameterized state value function: $V_\psi(s_t)$

- soft Q-function $Q_\theta(s_t, a_t)$

- tractable gaussian policy paremetrized by a neural network $\pi_\phi(a_t|s_t)$

- Loss for soft value function approximator training:
  $$
  J_V(\psi) = \mathbb{E}_{s_t \sim \mathcal{D}}[\frac12(V_\psi(s_t) - \hat{V}(s_t) )^2]
  $$
  where,
  $$
  \hat{V}(s_t) = \mathbb{E}_{a_t \sim \pi_{\phi}}[Q_\theta(s_t, a_t) - log \pi_\phi(a_t|s_t)]
  $$
  and $\mathcal{D}$ is the distribution of replay buffer from which state $s_t$ is sampled and action $a_t$ is sampled from the current policy for that state

- Loss for soft Q-function approximator training:
  $$
  J_Q(\theta) = \mathbb{E}_{(s_t, a_t) \sim \mathcal(D)}[\frac12(Q_\theta(s_t,a_t) - \hat{Q}(s_t, a_t))^2]
  $$
