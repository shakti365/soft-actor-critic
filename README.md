# Soft Actor-Critic
#### Implementation of Soft Actor-Critic Algorithm in TF2

Notes: https://shivamshakti.dev/posts/sac

## Usage

- Create a virtual environment for Python (I use [this](https://gist.github.com/shakti365/c8384d421ace17a6586f5b8733d5705c) setup)

- Install the dependencies

  ```
  pip install -r requirements.txt
  ```

- Run the training script

  ```
  cd src
  python main.py # Uses `MountainCarContinuous-v0` by default
  ```

- Run the evaluation script

  ```
  python play.py --model_name <PATH_TO_SAVED_MODEL>
  ```




## References

- [Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor](https://arxiv.org/pdf/1801.01290.pdf)
- [Soft Actor-Critic Algorithms and Applications](https://arxiv.org/pdf/1812.05905.pdf)
- [Learning to Walk via Deep Reinforcement Learning](https://arxiv.org/pdf/1812.11103.pdf)
- [Open AI SAC](https://spinningup.openai.com/en/latest/algorithms/sac.html)