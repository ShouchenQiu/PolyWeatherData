import os
for p in [r"d:\NN\PolyWeatherData\scripts\tmp_exhaustive_out.txt",
          r"d:\NN\PolyWeatherData\scripts\tmp_read2.txt"]:
    if os.path.exists(p):
        print(p, "size:", os.path.getsize(p))
    else:
        print(p, "MISSING")
