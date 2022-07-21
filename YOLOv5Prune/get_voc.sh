start=$(date +%s)
mkdir -p ../tmp
cd ../tmp/

# Download/unzip images and labels
d='.' # unzip directory
url=https://github.com/ultralytics/yolov5/releases/download/v1.0/
f1=VOCtrainval_06-Nov-2007.zip # 446MB, 5012 images
f2=VOCtest_06-Nov-2007.zip     # 438MB, 4953 images
for f in $f3 $f2 $f1; do
  echo 'Downloading' $url$f '...' 
  curl -L $url$f -o $f && unzip -q $f -d $d && rm $f & # download, unzip, remove in background
done
wait # finish background tasks

end=$(date +%s)
runtime=$((end - start))
echo "Completed in" $runtime "seconds"