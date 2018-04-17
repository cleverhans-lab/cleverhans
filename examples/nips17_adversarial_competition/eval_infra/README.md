
# Evaluation infrastructure for adversarial competition

## About

This directory contains code of the backend which was used to run evaluation of
[NIPS17 Adversarial Competition](https://www.kaggle.com/nips-2017-adversarial-learning-competition).

This file describes how to run and use the code.
For more details about the competition refer to
[our publication](https://arxiv.org/abs/1804.00097) and to
[Kaggle competition page](https://www.kaggle.com/nips-2017-adversarial-learning-competition).

NOTES AND DISCLAIMERS:

* **This code is provided AS IS. At the time of publishing this code
  was well tested and was working for evaluation of the competition.
  We do not provide any future support of this code.**

* This code is formatted according to
  [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
  which is different from PEP8 (biggest difference is number of spaces
  in indentation). We do not plan to reformat the code according to PEP8.

* You can defer questions about this code to
  [Alexey Kurakin](https://github.com/AlexeyKurakin), however they will
  be answered on best effort basis.

## How to run

During the evaluation all attacks are run on the image dataset to produce
adversarial images, then all defenses are run on all adversarial images to
produce classification labels and finally scores are computed based
on classification labels, true classes and target classes of all images.

Competition is evaluated in a distributed way by running multiple **workers**
on [Google Cloud Compute Engine](https://cloud.google.com/compute/).
At any given moment each worker runs single **attack work piece** - evaluation
of a single attack on a batch of clean images
or **defense work piece** - evaluation of a single defense on a batch of
adversarial images. All attack work pieces are independent from each
other and could be run in parallel. After all attack work pieces are done,
defense work pieces are run in parallel.

Coordination between workers is done via
[Google Cloud Datastore](https://cloud.google.com/datastore/) where list of
all work pieces is stored. Content of the datastore is populated by **master**
which could be run either on local machine or on Google Cloud.

So evaluation of the competition consists of following steps:

* Validate all submissions and copy valid submissions to Google Cloud.
* **Master** populates all attack work pieces.
* **Workers** are deployed to Google Cloud VMs.
* **Workers** compute all attack work pieces.
* **Master** populates all defense work pieces.
* **Workers** compute all defense work pieces.
* **Master** computes scores of all submissions.

Below this process is described in more details.

### 1. Preparation

Before running the competition you need to prepare machine which will run master
as well as Google Cloud VMs which will run workers.

#### 1.1 Prepare Google Cloud Project

Since most of the competition is run on Google Cloud you have to have
[Google Cloud project](https://cloud.google.com/resource-manager/docs/creating-managing-projects).
Also you need to create
[Google Cloud Storage bucket](https://cloud.google.com/storage/docs/creating-buckets)
for the competition data within your Google Cloud project.

You also will need to install Google Cloud SDK to use `gsutil` - a tool to copy
data to and from Google Cloud Storage.

#### 1.2 Prepare dataset

You need to prepare dataset of labeled images which will be used in the
competition.
You can either download DEV or FINAL dataset which was used in NIPS17
competition or prepare your own dataset in a similar format.
NIPS17 competition dataset could be found
[here](https://github.com/tensorflow/cleverhans/tree/master/examples/nips17_adversarial_competition/dataset).

To download NIPS17 dataset make a local copy of
[this](https://github.com/tensorflow/cleverhans/tree/master/examples/nips17_adversarial_competition/dataset)
directory and then run following commands:

```bash
# ${DATASET_DIR} is a directory where dataset will be downloaded
# ${DATASET_NAME} is a name of the dataset, "dev" or "final"

# Prepare directory with downloaded dataset
mkdir -p ${DATASET_DIR}/images

# Copy dataset metadata there
cp ${DATASET_NAME}_dataset.csv ${DATASET_DIR}

# Download dataset images
python download_images.py --input_file=${DATASET_NAME}_dataset.csv \
 --output_dir=${DATASET_DIR}/${DATASET_NAME}/images
```

After you done `${DATASET_DIR}` will contain dataset metadata and all dataset
images in `images` subdirectory.

Then copy the dataset into Google Cloud Storage:

```bash
# copy metadata
gsutil cp ${DATASET_DIR}/${DATASET_NAME}_dataset.csv \
  gs://${GOOGLE_CLOUD_STORAGE_BUCKET}/dataset/

# copy images
gsutil -m cp ${DATASET_DIR}/images/* \
  gs://${GOOGLE_CLOUD_STORAGE_BUCKET}/dataset/${DATASET_NAME}/
```

Keep the content of the `${DATASET_DIR}`,
you will need to copy it to the worker VMs in one of the next steps.

#### 1.3 Update config

Before proceeding with next steps you need to update variables in
config file `scripts/config.sh`.

Update `GOOGLE_CLOUD_PROJECT_ID` and `GOOGLE_CLOUD_STORAGE_BUCKET` with names
of Google Cloud Project ID and Google Cloud Storage Bucket which you will use
for competition.

Additionally consider updating `VIRTUALENV_NAME` if default value does not work
for you.

Few other variables will be updated in next steps.

#### 1.4 Preparation of master machine

Master has to be run only a few times to populate work pieces for workers and
to compute final scores. These tasks should take no more than few hours combined
(exact time depends on how fast your computer, network connection and how many
submissions are in the competition).
Thus master could be run on your own machine.
Below we call a computer where you runs master as **master machine**.

Generally speaking master machine can have any operating system,
however we only tested master on Debian flavor of Linux.
Also if you operating system does not support running bash scripts
you may need to rewrite a few bash scripts in a way which is supported on
your machine.

To prepare master machine you need to do following:

1. Install Python 2.7
2. Install [Google Cloud SDK](https://cloud.google.com/sdk/docs/)
2. Run `scripts/prepare_virtualenv.sh` to prepare python virtual environment
   for the master
3. Run `gcloud auth application-default login` or set
   `GOOGLE_APPLICATION_CREDENTIALS` to proper value. This is needed for master
   to be able to authenticate to Google Cloud services. See
   [Application Default Credentials](https://developers.google.com/identity/protocols/application-default-credentials)
   for more information.

#### 1.5 Copy baselines to Google Cloud Storage

Each baseline is a zip archive which name starts with `baseline_`.
This archive should be formatted in a same way as any other valid submission.

You can use sample attacks and defenses from
[dev toolkit](https://github.com/tensorflow/cleverhans/tree/master/examples/nips17_adversarial_competition/dev_toolkit)
as baselines. Just keep in mind that you have to download checkpoints for these
examples and package them into zip archives yourself.

After you prepared set of baselines copy them into following locations
in Google Cloud Storage:

* `gs://${GOOGLE_CLOUD_STORAGE_BUCKET}/baselines/defense` - directory with
  defense baselines
* `gs://${GOOGLE_CLOUD_STORAGE_BUCKET}/baselines/targeted` - directory with
  targeted attack baselines
* `gs://${GOOGLE_CLOUD_STORAGE_BUCKET}/baselines/nontargeted` - directory with
  non-targeted attack baselines

#### 1.6 Preparation of a snapshot for workers VMs

First of all you need to decide in which zone you will be creating all VMs
for the competition and update `GOOGLE_CLOUD_COMPUTE_ZONE` variable
in `config.sh` with the name of the zone.
Optionally you may want to update `GOOGLE_CLOUD_VM_USERNAME`
depending on which username you will use to log in into VMs.
After updating `config.sh` you will be able to use script `scp_cloud_vm.sh` to
copy files to and from VM.

You need to prepare snapshot of one VM manually, then you can use
helper script to create many workers from this snapshot.

To prepare snapshot:

1. Create a Cloud VM with attached GPU and Ubuntu 16.04 OS.

   * Make sure to enable read/write access to Google Cloud Storage for this VM.

   * If you're planning to pre-download all Docker images
     you may want to increase disk size of this machine to 100GiB or more.

2. Install some additional packages to VM:

   ```bash
   sudo apt-get update
   sudo apt-get upgrade
   # Install pip and virtualenv for Python 2
   sudo apt-get install python-pip
   sudo pip install --upgrade pip
   sudo pip install virtualenv
   # Install zip
   sudo apt-get install zip unzip
   ```

3. Install [Docker](https://docs.docker.com/install/linux/docker-ce/ubuntu/)
   into VM.
   * Make sure to follow "Manage Docker as a non-root user" section in
   [post installation steps](https://docs.docker.com/install/linux/linux-postinstall/)
   so you can run `docker` command without `sudo`

4. Install NVidia drivers and any other dependencies needed for NVidia Docker
   into VM.

   * You can get the latest version of the driver from
     [NVidia web-site](http://www.nvidia.com/object/unix.html)
     (look for the latest Linux x86_64 driver) and install it on your VM.
     You can safely ignore warnings about X library path and
     32-bit compatibility binary during the installation.

   * Please refer to [NVidia Docker](https://github.com/NVIDIA/nvidia-docker)
     documentation for details about any additional dependencies.

5. Install [NVidia Docker](https://github.com/NVIDIA/nvidia-docker) into VM.

6. Make sure that NVidia Docker working properly and can detect your graphic
   card. To do this run
   `docker run --runtime=nvidia --rm nvidia/cuda nvidia-smi` inside the VM.
   You should see output of NVIDIA-SMI tool which shows utilization of
   your graphic card. If it fails or can not detect your graphic card
   please resolve the problem before proceeding to next steps.

7. Copy dataset into VM. This step is run from outside of the VM.

   * At this point you should have dataset downloaded into `${DATASET_DIR}`
     So run `scripts/copy_dataset_to_vm.sh` to copy dataset to VM:

     ```bash
     # To copy dev dataset:
     scripts/copy_dataset_to_vm.sh ${DATASET_DIR} dev ${VM_NAME}
     # To copy final dataset:
     scripts/copy_dataset_to_vm.sh ${DATASET_DIR} final ${VM_NAME}
     ```

8. Copy evaluation infrastructure to VM. This step is done from outside
   of the VM by running following script:

   ```bash
   scripts/copy_eval_infra_to_vm.sh ${VM_NAME}
   ```

   NOTE: at this point don't worry if some of the values in `config.sh` are not
   updated. Script which starts workers will copy updated version of competition
   infrastructure with updated `config.sh`.

9. Run `eval_infra/scripts/prepare_virtualenv.sh` inside the VM to prepare
   python virtual environment needed to run the worker.

10. Shut down VM, make a snapshot of its disk (via Google Cloud web UI)
    and set value of `GOOGLE_CLOUD_REF_VM_SNAPSHOT` in `config.sh`
    to the name of the snapshot.

This snapshot will be used during actual evaluation of the competition to
create VMs with workers.




### 2. Running the competition

Evaluation infrastructure can be used to run multiple rounds of the
competition. Each round involves independent evaluation of all provided
submission on given dataset.

Intermediate data for each round are stored in separate
subdirectory `gs://${GOOGLE_CLOUD_STORAGE_BUCKET}/${ROUND_NAME}` in
Google Cloud Storage and using namespace `${ROUND_NAME}`
in Google Cloud Datastore. Thus intermediate data for each round have no
intersection with each other and you can run next round without cleaning up
data of previous round as long as names of the rounds are different.

Before you start evaluation of one round you need to decide which dataset you
will use and what will be the name of the round. Then update variables
`DATASET` and `ROUND_NAME` in `config.sh` on master machine (workers will
be updated later).

Also you may want to update `MASTER_RESULTS_DIR` and `NUM_DEFENSE_SHARDS`
if default values does not work for you.

#### 2.1 Validate and copy submissions and baselines

First step of running the evaluation is to copy submissions and baselines to
Google Cloud Storage directory for current evaluation round.

To copy baselines from `gs://${GOOGLE_CLOUD_STORAGE_BUCKET}/baselines/` into
directory which correspond to current round of the competition run following
command:

```bash
scripts/copy_baselines.sh
```

To validate submissions and copy them into appropriate location in Google Cloud
Storage you can use program `validation_tool/validate_and_copy_submissions.py`.
This tool scans all zip, tar and tar.gz archives in given Google Cloud
Storage location, detects which of them are valid submissions and copies them
to submission directory of the current round.

You can run this tool in the VM created at
"Preparation of a snapshot for workers VMs" step.
To do this, turn this VM on, ssh into it and run following commands:

```bash
source ~/.virtualenv/${VIRTUALENV_NAME}/bin/activate
cd eval_infra/validation_tool
# ${SOURCE_DIR} is a directory in Google Cloud Storage which contains archives
# with all submissions.
# validate_and_copy_submissions.py will scan all subdirectories of the
# ${SOURCE_DIR}, try to interpret all found zip, tar and tar.gz archives as
# submissions, validate them and copy archives which are valid submissions
# into target directory
python -B validate_and_copy_submissions.py --use_gpu \
  --source_dir=${SOURCE_DIR} \
  --target_dir=gs://${GOOGLE_CLOUD_STORAGE_BUCKET}/${ROUND_NAME}/submissions \
  --containers_file=${HOME}/containers.txt \
  --log_file=${HOME}/validation_log.txt
deactivate
```

As a side effect this command will also download all Docker images from
all submissions into the VM.
So you can re-create snapshot `${GOOGLE_CLOUD_REF_VM_SNAPSHOT}` of this VM
right now and this snapshot will contain all Docker images of all submissions.

Using snapshot with all Docker images will help you to save time and
network traffic during actual evaluation of the submissions.


#### 2.2 Prepare attack work pieces

To prepare attack work pieces and write them to Google Cloud Datastore
run following command:

```bash
run_master.sh attack
```
This operation may take several minutes so be patient.

If you interrupt preparation of the attack work pieces you probably will need
to clean up datastore before retrying.
To cleanup datastore run `run_master.sh cleanup_datastore`

NOTE: If you want to have a test run of your competition on a small subset
of the dataset then you can add flag `--limited_dataset` into `run_master.sh`
as additional argument to `master.py`. This flag will force to use only 30
images from the dataset with batch size 10.

#### 2.3 Create worker VMs

To avoid confusion let's introduce some terminology:

* **creating VM** - create virtual machine which will host worker.
* **starting VM** - start virtual machine which will be used to
  run worker. After worker VM is started we will say that
  **worker VM is up and running**.
* **starting worker** - start worker script which performs evaluation on a VM.
  After script is started we will say that **worker is running**.

At this step we only create and start VMs, but not the worker script.

At this point you need to decide how many workers you need and what will be
their indices. Indices could be any non-negative numbers, but for simplicity we
recommend to use 0, ... (N-1) as indices where N is total number of workers.

Depending on chosen number of workers you may want
to update `NUM_DEFENSE_SHARDS` in `config.sh`. We recommend to use
min(1, NUM_WORKERS/10) defense shards.

After you decided how many workers you need and what are their indices,
create worker VMs by running `scripts/create_workers.sh`:

```bash
# Create workers with indices ${INDICES}
# ${INDICES} should be list of numbers separated by spaces
scripts/create_workers.sh "${INDICES}"

# Create workers with indices 0, 1, 2, ..., 9
scripts/create_workers.sh "$(seq 0 9)"

# Create workers with indices 1, 3 and 5
scripts/create_workers.sh "1 3 5"
```

Script `create_workers.sh` only create VMs and does not start worker code
on these VMs. You need to wait until all VMs are up and running (if you can
ssh into VM then it's ready) and proceed to the next step to start worker code.

NOTE: Keep in mind that `scripts/create_workers.sh` creates VMs
in 'running' state and you will be billed for uptime of these machines.
You may want to shut them down when they are not in use to avoid paying for
idle machines.

#### 2.4 Starting and restarting workers

Before starting workers you need to make sure that all worker VMs are up and
running. Also you need to make sure that worker script is not already running
on the VMs.
If VM was shut down, then simply start it from Google Cloud web UI and wait
until Google Cloud will show that VM is up and running.
If VM is already running then reload it using Google Cloud web UI and
wait a few minutes until reload is complete.

After worker VMs are up and running you can start workers by using
script `scripts/start_workers.sh`:

```bash
# to start workers 0,1,2,3, ... 9
scripts/start_workers.sh "$(seq 0 9)"
# to start workers 1, 5 and 7:
scripts/start_workers.sh "1 5 7"
```

Sometimes worker get stuck or fail. In such case you may need to restart
failed workers. To do this restart corresponding VMs and then use
the same script `scripts/start_workers.sh` to start workers.

#### 2.5 Checking status of the workers

Any time you can run following command to monitor progress of competition
evaluation:

```bash
run_master.sh status
```

Note that this command may take several minutes to complete.

`run_master.sh status` will show how many attack and defense work pieces
are done and how many work pieces was finished by each worker. It also show
last time each worker completed a work piece.

If you notice that some workers haven't had any updates for a long time
it may indicate that they crash or stuck or some other problem have happen.
You can log in to the VM with stuck worker and check `log.txt` to troubleshoot
it, however in most cases these issues are transient and resolved by
restarting stuck worker.

#### 2.6 Prepare defense work pieces

After all attack work pieces are done (according to `run_master.sh status`)
you need to populate defense work pieces.
To populate defense work pieces run following command:

```bash
run_master.sh defense
```

This command may take many minutes to complete, so be patient.

If for some reason you ended up with corrupted state of the database you can
cleanup defense work pieces by running `run_master.sh cleanup_defenses`.

#### 2.7 Compute results

As soon as all attack work pieces and defense work pieces are evaluated you
can shut down VMs with all workers (to avoid paying for idle VMs).
Then you can run following command to compute results:

```bash
run_master.sh results
```

It may take several minutes to compute all results. When done results will
be saved into `${MASTER_RESULTS_DIR}` directory (as specified in `config.sh`)
on master machine.

Results include following files:

* `accuracy_matrix.csv` - matrix with the number of correctly classified
  adversarial images for each pair of defense and attack (both targeted
  and non-targeted). Rows are defenses, columns are attacks.
* `error_matrix.csv` - matrix with the number of misclassified adversarial
  images for each pair of defense and attack (both targeted and non-targeted).
  Rows are defenses, columns are attacks.
* `hit_target_class_matrix.csv` - matrix with the number of times a defense
  recognized adversarial images produced by an attack
  (both targeted and non-targeted) as specific target class.
  Rows are defenses, columns are attacks.
* `attack_results.csv` and `sorted_attack_results.csv` - unsorted and sorted
  results of non-targeted attacks. Sorted results are sorted by score
  and contain less stats about each submission.
* `targeted_attack_results.csv` and  `sorted_target_attack_results.csv` -
  unsorted and sorted results of targeted attacks.
* `defense_results.csv` and `sorted_defense_results.csv` - unsorted and sorted
  results of defenses.
* `attack_id_to_name.csv` - mapping from internal IDs of all targeted and
  non-targeted attacks to their external IDs. Internal IDs are used only
  by evaluation infrastructure, external IDs is what was provided as inputs.
* `defense_id_to_name.csv` - mapping from internal IDs of all defenses
  to their external IDs.
* `count_adv_images.csv` - file with number of adversarial images produces by
  each attack.
* `classified_images_count.csv` - file with total number of images classified
  by each defense.
* `*.dump` - these are temporary files with intermediate results
  saved by master. If master interrupted during computation of final results and
  then restarted these files are used to resume execution of the master
  without recomputing too much data.

Additionally master will save log `log.txt` into the same directory.


## Known issues

### Python 3 compatibility

Most of the code is compatible with both Python 2 and Python 3
However we haven't perform end-to-end testing of the entire competition
evaluation with Python 3 thus we recommend to run code using Python 2.

### Same attack work piece evaluated twice

This issue happens very rare with the current values of worker parameters,
nevertheless its possible theoretically
and below is an instruction on how to deal with it.

It's possible that the same attack work piece will be evaluated twice.
It can happen in following circumstances:

* Worker A started evaluation of work piece W1, however got stuck downloading
  Docker container.
* Worker B saw that work piece W1 was claimed by A too long ago, concluded that
  worker A crashed or got rebooted and started evaluation of work piece W1
* Worker A finished evaluation of work piece W1, wrote list of produced
  adversarial images to datastore, wrote archive with adversarial images to
  Google Cloud Storage. Worker A tried to mark work piece as completed
  but failed because it's already re-claimed by B. Nevertheless adversarial
  images were already written to datastore and storage.
* Worker B finished evaluation of work piece W1, overwrote list of produced
  adversarial images in the datastore and overwrote archive with adversarial
  images into storage. Then worker B marked work piece W1 as completed.

This behavior is by design and generally speaking does not cause any issues
as long as in both cases result of evaluation of work piece W1 is the same.

However if work piece W1 times out and does not output adversarial images
for all inputs, it's possible that two different evaluations of the work piece
will result in different number of produced images.

In such case it's possible that archive with adversarial images written to
Google Cloud Storage will contain less images than listed in Google Cloud
Datastore.

If this happen then defenses will classify less adversarial images than
listed in the datastore.
Then during computation of the final scores by master
this inconsistency will lead to the fact that no defenses will be used to
compute attack scores, so all attacks will receive score 0.

The issue could be diagnosed by the fact that all attacks have score zero
and following line appear in the log output of the master:

```
Number of defenses to use to score attacks: 0
```

If issue have happened and you discovered it during computation of results
then following could be done to fix it:

* Manually find problematic piece of attack work and edit the content of the
  datastore to fix the issue. Then you can delete all `*.dump` files are
  redo computation of results by master. This is a clean way to fix the problem,
  but it might be time consuming.
* Change `total_num_adversarial` in `EvaluationMaster.compute_results` in
  `master.py` to be equal to maximum value of `classified_images_count` dict
  (or maximum value from `classified_images_count.csv` file). Then again
  delete all `*.dump` files and restart master.
  This is a quick and hacky way to fix the issue, but it still should work
  in most cases.

Moreover following steps could be done to decrease probability
of the issue happening in the first place:

* Pre-download all Docker images to each worker VM, so workers won't spend time
  downloading Docker image before running new submission.
  However if evaluation of competition takes multiple days and some of the
  submissions refer to the "latest" version of some Docker image
  (e.g. tensorflow/tensorflow:latest-gpu) instead of fixed version it's
  possible that the latest version of Docker image will be updated during
  evaluating of the competition and some workers will download it again.
* Increase `MAX_PROCESSING_TIME` in `work_data.py`. This constant define
  how long worker is allowed to process one piece of work before it considered
  failed.
