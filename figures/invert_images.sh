# iterate all *.png files in current directory
for file in *_light.png; do
  magick convert "$file" -channel RGB -negate "${file/_light.png/_dark.png}"
done
