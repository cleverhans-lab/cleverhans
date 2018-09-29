#!/bin/sh
set -e
# run keras backend init to initialize backend config
python -c "import keras.backend"
# create dataset directory to avoid concurrent directory creation at runtime
mkdir ~/.keras/datasets
# set up keras backend
sed -i -e 's/"backend":[[:space:]]*"[^"]*/"backend":\ "'$KERAS_BACKEND'/g' ~/.keras/keras.json;
echo -e "Running tests with the following config:\n$(cat ~/.keras/keras.json)"
# test for evaluation infrastructure are very fast, so running them first
nosetests -v --nologcapture -w examples/nips17_adversarial_competition/eval_infra/code/ eval_lib/tests/
# --nologcapture: avoids a large amount of unnecessary tensorflow output
# --stop: stop on first error. Gets feedback to Cloud Build faster
if [[ "$PYTORCH" == True ]]; then
  nosetests --nologcapture -v --stop tests_pytorch;
elif [[ "$PYTORCH" == False ]]; then
  nosetests -v --nologcapture --stop cleverhans;
  nosetests --nologcapture -v --stop tests_tf;
fi
