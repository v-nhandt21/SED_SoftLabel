
from pathlib import Path

def gen_dev_txt_scores_gt():
     with open("/home/nhandt23/Desktop/DCASE/Wav2Vec/metadata/soft_gt_dev.csv", "r", encoding="utf-8") as f:
          lines = f.read().splitlines()
          meta = lines[0]
          file, meta = meta.split("\t",1)
          for line in lines[1:]:
               file, out = line.split("\t",1)
               file = file.replace(".wav",".tsv")
               if Path("dev_txt_scores_gt/"+ file).is_file() == False:
                    fw = open("dev_txt_scores_gt/"+ file, "a+", encoding="utf-8")
                    fw.write(meta+"\n")
               else:
                    fw = open("dev_txt_scores_gt/"+ file, "a+", encoding="utf-8")
                    fw.write(out+"\n")
                    
gen_dev_txt_scores_gt()