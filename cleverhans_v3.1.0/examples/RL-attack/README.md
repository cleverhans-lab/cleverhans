# Crafting Adversarial Example Attacks on Policy Learners 
Examples of test-time and training-time FGSM attacks on Deep Q-Networks (DQNs). These examples are based on RLAttack ( https://github.com/behzadanksu/rl-attack/ ), a framework that provides an interface between Cleverhans and [@openai/baselines](https://github.com/openai/baselines) for experimental analysis of adversarial example attacks on policy learning in Deep RL. Attack models and methodologies are detailed in our paper "Whatever Does Not Kill Deep Reinforcement Learning, Makes It Stronger" (Behzadan & Munir, 2017 - https://arxiv.org/abs/1712.09344 ). 

### Dependencies
- Python 3
- cleverhans v2.0.0
- RLAttack v1.0.0

```
git clone https://github.com/behzadanksu/rl-attack.git
cd rl-attack
pip install -e .
```

### Examples
Two example scripts are included. 

- enjoy-adv.py : a sample implementation of test-time FGSM attack on pre-trained DQN Atari agents.
- train.py: a sample implementation of training-time FGSM attack on DQN Atari agents.

Some example executions on the Breakout game environment are: 

- Test-time, No attack, testing a DQN model of Breakout trained without parameter noise:

```
$> python3 enjoy-adv.py --env Breakout --model-dir ./data/Breakout/model-173000 --video ./Breakout.mp4
```

- Test-time, No attack, testing a DQN model of Breakout trained with parameter noise (NoisyNet implementation):
```
$> python3 enjoy-adv.py --env Breakout --noisy --model-dir ./data/Breakout/model-173000 --video ./Breakout.mp4
```

- Test-time, Whitebox FGSM attack, testing a DQN model of Breakout trained without parameter noise:
```
$> python3 enjoy-adv.py --env Breakout --model-dir ./data/Breakout/model-173000 --attack fgsm --video ./Breakout.mp4
```

- Test-time, Whitebox FGSM attack, testing a DQN model of Breakout trained with parameter noise (NoisyNet implementation):
```
$> python3 enjoy-adv.py --env Breakout --noisy --model-dir ./data/Breakout/model-173000 --attack fgsm --video ./Breakout.mp4
```

- Test-time, Blackbox FGSM attack, testing a DQN model of Breakout trained without parameter noise:
```
$> python3 enjoy-adv.py --env Breakout --model-dir ./data/Breakout/model-173000 --attack fgsm --blackbox --model-dir2 ./data/Breakout/model-173000-2 --video ./Breakout.mp4
```

- Test-time, Blackbox FGSM attack, testing a DQN model of Breakout trained with parameter noise (NoisyNet implementation), replica model trained without parameter noise:
```
$> python3 enjoy-adv.py --env Breakout --noisy --model-dir ./data/Breakout/model-173000 --attack fgsm --blackbox --model-dir2 ./data/Breakout/model-173000-2 --video ./Breakout.mp4
```

- Test-time, Blackbox FGSM attack, testing a DQN model of Breakout trained with parameter noise (NoisyNet implementation), replica model trained with parameter noise:
```
$> python3 enjoy-adv.py --env Breakout --noisy --model-dir ./data/Breakout/model-173000 --attack fgsm --blackbox --model-dir2 ./data/Breakout/model-173000-2 --noisy2 --video ./Breakout.mp4
```

- Training-time, Whitebox attack, no parameter noise, injecting adversarial example with 20% probability:

```
$> python3 train.py --env Breakout --save-dir ./data/Breakout/ --attack fgsm --num-steps 200000000 --attack-prob 0.2 
```

- Training-time, Whitebox attack, NoisyNet parameter noise, injecting adversarial example with 100% probability:

```
$> python3 train.py --env Breakout --noisy --save-dir ./data/Breakout/ --attack fgsm --num-steps 200000000 --attack-prob 1.0
```
