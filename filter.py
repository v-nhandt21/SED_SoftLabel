import glob, os

for exp in glob.glob("Outdir/*"):
     check = 0
     for fi in glob.glob(exp+"/*"):
          if "fold" in fi: 
               check += 1
     if check >= 5:
          print(exp)
          
     else:
          os.system("rm -r "+exp)