## CleverHans Example Submission to the Robust Vision Benchmark

Using the wrappers in this repository, submitting a CleverHans attack to the [Robust Vision Benchmark](https://robust.vision/benchmark) requires just a few lines of code:

```python
#!/usr/bin/env python3

import numpy as np
from cleverhans.attacks import FastGradientMethod
from robust_vision_benchmark import attack_server
from utils import cleverhans_attack_wrapper


def attack(model, session, a):
    fgsm = FastGradientMethod(model, sess=session)
    image = a.original_image[np.newaxis]
    return fgsm.generate_np(image)


attack_server(cleverhans_attack_wrapper(attack))
```

The full example can be found in the `cleverhans_attack_example` folder.

### Testing an attack

Just install the latest version of the [robust-vision-benchmark python package](https://github.com/bethgelab/robust-vision-benchmark) using

```bash
pip install --upgrade robust-vision-benchmark
```

and run

```bash
rvb-test-attack cleverhans_attack_example/
```

to test the attack. Once the test succeeds, you can **[submit your attack](https://github.com/bethgelab/robust-vision-benchmark)**.
