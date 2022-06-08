set -e 

MODE=$1
SPEEDTEST=$2


cd src/

out_path="../out/"
if [ -d "$out_path" ]; then
    rm -rf "$out_path"
else
    echo "file $out_path is not exist."
fi

mkdir -p "$out_path"


if [ ${MODE} = "image" ]; then
	python image.py
elif [ ${MODE} = "video" ]; then
    python video.py ${SPEEDTEST}
else
    echo -e "The mode must be image or video"
fi

exit 0