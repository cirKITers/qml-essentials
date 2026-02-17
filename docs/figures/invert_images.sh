find . -type f -name '*_light.png' -print0 | while IFS= read -r -d '' file; do
  out="${file%_light.png}_dark.png"
  tmp="${out}.tmp.$$"

  # create converted image in a temporary file
  magick "$file" -transparent white -channel RGB -negate "$tmp" || {
    rm -f "$tmp"
    continue
  }

  if [ -f "$out" ]; then
    # compare existing and new; if identical, discard temp
    if diff -q "$out" "$tmp" >/dev/null 2>&1; then
      rm -f "$tmp"
      continue
    fi
  fi

  # move new file into place (overwriting only when different or not present)
  mv -f "$tmp" "$out"
done