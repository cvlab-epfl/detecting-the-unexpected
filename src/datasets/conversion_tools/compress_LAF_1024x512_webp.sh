# Compress the LAF dataset from original 2048x1024 PNG to 1024x512 WEBP
# $DIR_LAF = location of original dataset
# $DIR_LAF_SMALL = output location

#DIR_LAF=/cvlabdata1/cvlab/dataset_LostAndFound
#DIR_LAF_SMALL=/cvlabsrc1/cvlab/dataset_LostAndFound/1024x512_webp

python compress_images.py \
	$DIR_LAF/leftImg8bit \
	$DIR_LAF_SMALL/leftImg8bit \
	"cwebp {src} -o {dest} -q 90 -sharp_yuv -m 6 -resize 1024 512" \
	--ext ".webp" --concurrent 20

python compress_images.py \
	$DIR_LAF/gtCoarse \
	$DIR_LAF_SMALL/gtCoarse \
	"convert {src} -filter point -resize 50% {dest}" \
	--ext ".png" --concurrent 20
