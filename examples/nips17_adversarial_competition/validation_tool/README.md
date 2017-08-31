
# Submission validation tool

This tool verifies that a submission file is valid or reports an error.
It extracts the submission, verifies presence and validity of metadata and runs
the submission on sample data.

Usage is following:

```bash
# FILENAME - filename of the submission
# TYPE - type of the submission, one of the following without quotes:
#   "attack", "targeted_attack" or "defense"
# You can omit --usegpu argument, then submission will be run on CPU
python validate_submission.py \
  --submission_filename=FILENAME \
  --submission_type=TYPE \
  --usegpu
```

After run this tool will print whether submission is valid or not.
If submission is invalid then log messages will contain explanation why.

NOTE: This tool might not work properly if the file is located on NFS or if
the directory containing submission file can't be mounted in Docker using `-v`
command line argument. In such case copy file to different location and try
again.
