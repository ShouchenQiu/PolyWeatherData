import codecs

src = r"d:\NN\PolyWeatherData\scripts\tmp_series_sweep_out.txt"
dst = r"d:\NN\PolyWeatherData\scripts\tmp_series_sweep_utf8.txt"

with codecs.open(src, "r", "utf-16") as f:
    text = f.read()
with codecs.open(dst, "w", "utf-8") as f:
    f.write(text)
print("converted", len(text), "chars")
