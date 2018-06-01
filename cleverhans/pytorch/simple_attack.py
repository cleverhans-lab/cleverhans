
class PGDAttack:
  def __init__(self, pytorch_model, epsilon=8.0, num_steps=10, step_size=2.0, random_start=False, loss_func='cw'):
    """Attack parameter initialization. The attack performs k steps of
       size a, while always staying within epsilon from the initial
       point."""
    pass

  def generate(self, x_nat, y):
    """Given a set of examples (x_nat, y), returns a set of adversarial
       examples within epsilon of x_nat in l_infinity norm."""
    pass



model = torch.models.ResNet()
attack = PGDAttack(model)

x_nat = get_data()
preds_nat = model(x_nat)

x_adv = attack.generate(x_nat, np.zeros(len(x_nat)))
preds_adv = model(x_nat)