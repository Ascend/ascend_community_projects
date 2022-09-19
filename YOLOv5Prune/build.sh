set -e
# Simple log helper functions
info() { echo -e "\033[1;34m[INFO ][MxStream] $1\033[1;37m" ; }
warn() { echo >&2 -e "\033[1;31m[WARN ][MxStream] $1\033[1;37m" ; }

rm -rf ./build
# complie
cmake -S . -Bbuild
make -C ./build  -j
echo "build done"