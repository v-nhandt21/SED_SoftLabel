import config 
from matplotlib import pyplot as plt

def addlabels(plt, x,y):
     for i in range(len(x)):
          plt.text(i, y[i], y[i], ha = 'center', color="red")
          
def plot_dataset():
     gt = open("metadata/gt_dev.csv", "r", encoding="utf-8")
     soft_gt = open("metadata/soft_gt_dev.csv", "r", encoding="utf-8")

     gt = gt.read().splitlines()[1:]
     soft_gt = soft_gt.read().splitlines()[1:]

     GT = [0]*11
     for i in gt:
          label = i.split("\t")[-1]
          GT[config.class_labels_hard[label]] += 1
          
     print(GT)

     GT_soft = [0]*11
     for i in soft_gt:
          label = i.split("\t")[3:]
          for id, l in enumerate(label):
               GT_soft[id] += float(l) 
               
     GT_soft = [int(z) for z in GT_soft]
     print(GT_soft)

     minus = []
     for x,y in zip(GT, GT_soft):
          if x<y:
               minus.append(x) 
          else:
               minus.append(0)
          
     print(minus)

     
     # function to add value labels
     
     # plt.figure().set_figwidth(40)
     # plt.figure().set_figheight(10)
     plt.rcParams['figure.figsize'] = [15, 12]

     labels = config.class_labels_hard.keys()

     plt.bar(labels, GT, color="darkorange")
     addlabels(plt, labels, GT)
     plt.bar(labels, GT_soft, color="green")
     addlabels(plt, labels, GT_soft)
     plt.bar(labels, minus, color="darkorange")

     y_pos = range(len(labels))
     plt.xticks(y_pos, config.class_labels_hard, rotation=45, ha='right')

     # legend
     handles = [plt.Rectangle((5,5),1,1, color="green"), plt.Rectangle((0,0),1,1, color="darkorange")]
     plt.legend(handles, ["soft_label", "hard_label"])

     plt.title("Label distribution for hard and soft label")

     plt.savefig("metadata/gt.png")
     # plt.show()

def plot_test():
     f = open("Outdir/meta_all.tsv", "r", encoding="utf-8")
     lines = f.read().splitlines()
     
     Test = [0]*11
     for line in lines:
          if line in config.class_labels_hard:
               i = config.class_labels_hard[line]
               Test[i] = Test[i] + 1
     
     plt.rcParams['figure.figsize'] = [15, 10]

     print(Test)
     print(sum(Test))
     
     labels = config.class_labels_hard.keys()
     y_pos = range(len(labels))
     plt.xticks(y_pos, config.class_labels_hard, rotation=45, ha='right')
     
     plt.bar(labels, Test, color="deepskyblue")
     
     plt.savefig("Outdir/test.png")
     
if __name__ == '__main__':
     # plot_dataset()
     plot_test()