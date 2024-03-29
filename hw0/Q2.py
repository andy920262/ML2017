import sys
from PIL import Image

file1 = Image.open(sys.argv[1])
file2 = Image.open(sys.argv[2])

w, h = file1.size
out = Image.new("RGBA", (w, h),(0, 0, 0, 0))
for x in range(w):
    for y in range(h):
        if file1.getpixel((x, y)) == file2.getpixel((x, y)):
            out.putpixel((x, y), (0, 0, 0, 0))
        else:
            out.putpixel((x, y), file2.getpixel((x, y)))
out.save('./ans_two.png')
