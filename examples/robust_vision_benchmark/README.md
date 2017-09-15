## CleverHans Example Submission to the Robust Vision Benchmark

Using the wrappers in this repository, submitting a CleverHans attack to the [Robust Vision Benchmark](https://robust.vision/benchmark) requires just [a few lines of code](cleverhans_attack_example/main.py). The full example can be found in the `cleverhans_attack_example` folder.

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
