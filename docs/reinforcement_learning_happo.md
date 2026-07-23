# Reinforcement Learning and HAPPO

This document explains the reinforcement-learning layer used in OmniSearch. It
focuses only on HAPPO training. The reward terms, observation vector, and
communication-dropout experiments are described separately, because those are
environment-design choices rather than the training algorithm itself.

## 1. Reinforcement Learning View

OmniSearch is a sequential decision problem. At each simulation step, every
agent receives an observation, chooses a continuous 2D movement action, and then
the simulator advances the wildfire search-and-rescue mission.

For agent $i$ at time $t$:

$$
o_{i,t} \rightarrow a_{i,t} \rightarrow r_{i,t+1}, o_{i,t+1}
$$

The training objective is to learn policies that maximize expected discounted
return:

$$
G_t = \sum_{k=0}^{T-t-1} \gamma^k r_{t+k+1}
$$

Here $r_{t+1}$ is the reward after the next simulator step, $\gamma$ is the
discount factor, and $T$ is the episode horizon. OmniSearch uses
$\gamma = 0.99$ in the HAPPO configuration, so near-future rewards matter most,
but later mission outcomes still affect learning.

## 2. Policy, Value, Q-Function, and Advantage

The **policy** is the actor network. It maps an agent observation to an action
distribution:

$$
\pi_i(a_i \mid o_i)
$$

At evaluation time, OmniSearch usually uses the deterministic action implied by
the trained actor. During training, actions are sampled from the distribution so
the policy can explore.

The **Q-function** is the expected long-term value of taking an action and then
continuing with the current policy:

$$
Q^\pi(s, a) =
\mathbb{E}_\pi
\left[
\sum_{k=0}^{T-t-1} \gamma^k r_{t+k+1}
\mid S_t=s, A_t=a
\right]
$$

This is useful for intuition: a movement can be good even if the immediate
reward is small, as long as it creates later mission value.

HAPPO in OmniSearch does **not** learn an explicit Q-table. It uses a critic
that estimates the state value:

$$
V^\pi(s) =
\mathbb{E}_\pi
\left[
\sum_{k=0}^{T-t-1} \gamma^k r_{t+k+1}
\mid S_t=s
\right]
$$

The critic is used to estimate the **advantage**:

$$
A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)
$$

The advantage asks whether an action was better or worse than the policy's
usual behavior in the same situation. Positive advantage increases the
probability of similar actions; negative advantage decreases it.

In practice, OmniSearch uses Generalized Advantage Estimation (GAE):

$$
\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)
$$

$$
\hat{A}_t = \sum_{\ell=0}^{T-t-1}(\gamma\lambda)^\ell \delta_{t+\ell}
$$

The HAPPO configuration uses $\lambda = 0.95$, which balances low-variance
critic estimates with longer-horizon reward information.

## 3. What HAPPO Means

HAPPO stands for **Heterogeneous-Agent Proximal Policy Optimization**. It is a
multi-agent extension of PPO designed for settings where agents need not be
identical.

That matters for OmniSearch because the team is heterogeneous:

- UAVs scout from above and reason about aerial perception.
- UGVs move on terrain and confirm survivors on the ground.
- The team objective couples both roles, because a survivor must first become
  known and then be reached.

HAPPO is an actor-critic method:

- **Actors:** decentralized policies that choose each agent's action from its
  own observation.
- **Critic:** a centralized value estimator used only during training.
- **Advantage estimate:** the learning signal that tells each actor whether its
  sampled action helped the future return.
- **PPO clipping:** a trust-region-style update that prevents the policy from
  changing too aggressively after one batch of rollouts.

The basic PPO update uses the probability ratio

$$
\rho_t(\theta) =
\frac{\pi_\theta(a_t \mid o_t)}
{\pi_{\theta_\mathrm{old}}(a_t \mid o_t)}
$$

and optimizes the clipped surrogate objective

$$
L^\mathrm{PPO}(\theta) =
\mathbb{E}_t
\left[
\min
\left(
\rho_t(\theta)\hat{A}_t,\,
\mathrm{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t
\right)
\right]
$$

OmniSearch uses $\epsilon = 0.2$, so the actor update is constrained to a
moderate step away from the policy that collected the rollout data.

## 4. Centralized Training, Decentralized Execution

OmniSearch follows the common centralized-training/decentralized-execution
pattern.

During execution, each actor receives only its own agent observation:

$$
a_{i,t} \sim \pi_i(\cdot \mid o_{i,t})
$$

During training, the critic receives a shared training state. In the current
HARL adapter this shared state is the concatenation of all agents' local
observations:

$$
s^\mathrm{critic}_t =
[o_{1,t}, o_{2,t}, \ldots, o_{N,t}]
$$

The critic therefore has broader context when estimating whether the joint
behavior improved the episode return. The actor does not get that privileged
critic input at deployment time.

This separation is important in OmniSearch:

- The UAV may need to scout before a UGV can act meaningfully.
- The UGV can be idle early in an episode while waiting for a survivor signal.
- A local action can look unimportant immediately but still enable later team
  progress.

The centralized critic reduces learning noise by evaluating actions in the
joint team context, while the learned policies remain usable as decentralized
agent controllers.

## 5. HAPPO's Sequential Multi-Agent Update

The main HAPPO idea is that agents are updated sequentially rather than all at
once. After one actor update, the learning signal for the next actor is adjusted
to account for the policy change that already happened.

Conceptually:

1. Collect trajectories with the current joint policy.
2. Estimate returns and advantages with the centralized critic.
3. Update one agent or policy group with a PPO-style clipped objective.
4. Carry an update factor into the next agent or policy group.
5. Repeat until all actors have been updated.

This is the practical version of HAPPO's heterogeneous-agent trust-region idea:
each policy update is kept local and conservative, while the joint update is
structured so one actor's change does not invalidate the next actor's learning
signal.

## 6. OmniSearch Implementation

The HAPPO training path is implemented through HARL:

- [scripts/train_happo_smoke.py](../scripts/train_happo_smoke.py) builds the
  HAPPO configuration and command-line interface.
- [agents/harl_vec_env.py](../agents/harl_vec_env.py) exposes the batched VMAS
  wildfire environment as a HARL vectorized environment.
- [agents/harl_env.py](../agents/harl_env.py) documents the single-environment
  HARL contract.
- [agents/harl_runner.py](../agents/harl_runner.py) registers the OmniSearch
  environment with HARL and extends the runner with diagnostics, policy-group
  sharing, split critics, and warm-start loading.
- [agents/happo_policy.py](../agents/happo_policy.py) loads saved HAPPO actor
  checkpoints for evaluation and trajectory export.

The main HAPPO configuration uses:

| Quantity | Value |
|---|---:|
| Algorithm | HAPPO |
| Default smoke steps | 2,000 |
| Default research steps | 400,000 |
| Default smoke episode length | 150 steps |
| Default diagnostic episode length | 300 steps |
| Default research episode length | 500 steps |
| Rollout threads | CLI-configurable; diagnostic presets use 8 |
| Discount $\gamma$ | 0.99 |
| GAE $\lambda$ | 0.95 |
| PPO clip $\epsilon$ | 0.2 |
| Hidden sizes | 256, 256 by default in the CLI training script |
| Value normalization | enabled |
| Recurrent policy | configurable |

For joint UAV/UGV runs, OmniSearch can share actor parameters by agent class:
all UAVs use one UAV policy and all UGVs use one UGV policy. This preserves role
specialization while improving sample efficiency. The code also supports split
critics by class, so UAV and UGV advantages can be estimated with role-specific
critic buffers while still using joint training context.

## 7. Pretraining and Joint Fine-Tuning

OmniSearch can train role-specialized actors separately and then use those
weights to initialize joint training:

1. Train a UAV policy on the aerial scouting task.
2. Train a UGV policy on the ground-confirmation task.
3. Start the joint UAV+UGV HAPPO run from those actor checkpoints.
4. Initialize the joint critic fresh, because the separate pretraining critics
   do not represent the coupled team task.

This is useful because UGV learning is especially noisy in the full task. Early
in an episode, ground robots may be idle or weakly supervised while waiting for
UAV scouting information. Separate pretraining gives each role a reasonable
initial behavior before the harder joint credit-assignment problem is introduced.

The relevant command-line options are:

```bash
python scripts/train_happo_smoke.py \
  --share-param-by-agent-class \
  --warmstart-uav-model-dir /path/to/uav/models \
  --warmstart-ugv-model-dir /path/to/ugv/models
```

Only actor weights are loaded from the warm-start checkpoints. The joint critic
and value normalizer start from scratch.

## 8. Relation to Rewards, Observations, and Dropout

HAPPO is the optimizer. It does not define what the mission values are. The
environment supplies those through:

- the reward signal returned after each step,
- the observation vector available to each actor,
- the communication and information-sharing regime used in a run.

Those components should be read as the task specification. HAPPO then learns
policies that maximize the resulting long-term return under that specification.

## References

- Jakub G. Kuba, Ruiqing Chen, Muning Wen, Ying Wen, Fanglei Sun, Jun Wang, and
  Yaodong Yang. [Trust Region Policy Optimisation in Multi-Agent Reinforcement
  Learning](https://arxiv.org/abs/2109.11251), 2021.
- Zhongyao Li, Haotian Hao, Boqiang Ding, et al.
  [Heterogeneous-Agent Reinforcement Learning](https://jmlr.org/papers/volume25/23-0488/23-0488.pdf),
  JMLR 2024.
- PKU-MARL. [HARL official implementation](https://github.com/PKU-MARL/HARL).
- John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg
  Klimov. [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347),
  2017.
- John Schulman, Philipp Moritz, Sergey Levine, Michael Jordan, and Pieter
  Abbeel. [High-Dimensional Continuous Control Using Generalized Advantage
  Estimation](https://arxiv.org/abs/1506.02438), 2015.
