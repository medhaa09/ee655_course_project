#Usage- To run the Neural Style Transfer script, use the following command. You can modify the arguments to point to different content and style images.
``
python main.py \
  --content data/content/person1.jpg \
  --style data/style/starry_night.jpg \
  --output_dir outputs/run1 \
  --image_size 256 \
  --steps 250`
  ``