find . -type f -name '*_light.png' -print0 | while IFS= read -r -d '' file; do
  out="${file%_light.png}_dark.png"
  tmp="${out}.tmp.$$"

  # create the inverted image in a temporary file
  magick "$file" -transparent white -channel RGB -negate "$tmp" || {
    rm -f "$tmp"
    continue
  }

  if [ -f "$out" ]; then
    # compare raw pixel data only (strip all metadata via -strip + RGBA stream)
    hash_old=$(magick "$out" -strip RGBA:- 2>/dev/null | sha256sum | awk '{print $1}')
    hash_new=$(magick "$tmp" -strip RGBA:- 2>/dev/null | sha256sum | awk '{print $1}')
    if [ "$hash_old" = "$hash_new" ]; then
      echo "$file -> $out (identical pixels, skipping)"
      rm -f "$tmp"
      continue
    fi
  fi

  echo "$file -> $out"
  mv -f "$tmp" "$out"
done