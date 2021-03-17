#!/bin/bash
#
# Helper script which creates a tmux session, starts workers and opens log file
# inside this session.
# Should be only run automatically by start_workers.sh script
#

WORKER_ID=$1

if [ -z ${WORKER_ID} ]; then
  echo "Worker ID has to be provided"
  exit 1
fi

cd ~/

tmux new-session -s run -d
tmux rename-window -t run:0 "run"
tmux new-window -t run:1 -n "view"
sleep 2
tmux send-keys -t run:0 "eval_infra/code/run_worker_locally.sh ${WORKER_ID}" C-m
sleep 2
tmux send-keys -t run:1 "tail -f log.txt" C-m
