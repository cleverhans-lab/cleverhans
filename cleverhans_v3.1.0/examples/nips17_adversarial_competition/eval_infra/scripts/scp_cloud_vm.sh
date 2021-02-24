#!/bin/bash
#
# Helper script which copies files to and from VM
# Usage:
#   scp_cloud_vm.sh SRC DST
#

# Import config.sh
source "$( dirname "${BASH_SOURCE[0]}" )/config.sh"

# Copy files using function from config.sh
scp_cloud_vm $1 $2
