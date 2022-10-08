set -e

CUR_PATH=$(cd "$(dirname "$0")" || { warn "Failed to check path/to/run.sh" ; exit ; } ; pwd)

# Simple log helper functions
info() { echo -e "\033[1;34m[INFO ][MxStream] $1\033[1;37m" ; }
warn() { echo >&2 -e "\033[1;31m[WARN ][MxStream] $1\033[1;37m" ; }

. /usr/local/Ascend/ascend-toolkit/set_env.sh
. ../../MindX_SDK/mxVision-2.0.4/set_env.sh

python3.9 test.py
exit 
