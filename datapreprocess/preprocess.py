import os
import csv

def extract_true_commonsense(dataset="train"):
    infile_path = os.path.join("./SemEval2020-Task4/ALL data/"+dataset+".csv")
    os.makedirs("rawdata", exist_ok=True)
    outfile_path=os.path.join("rawdata/"+dataset+".txt")
    true_cs = []
    with open(infile_path, "r") as f:
        reader = csv.reader(f)
        count = 0
        for row in reader:
            if count == 0:
                pass
            else:
                true_cs.append("".join(row[0].split(".")).lower())
            count += 1
    with open(outfile_path, "w") as f:
        f.write("\n".join(true_cs))


def csv2dir(csv_path="raw_data/movie/movie_0_1_ac.csv", out_dir="dataset/movie/ac"):
    covers = []
    stegos = []
    os.makedirs(out_dir, exist_ok=True)
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        count = 0
        for row in reader:
            if count == 0:
                pass
            else:
                label = int(row[1])
                if label == 0:
                    covers.append(row[0])
                else:
                    stegos.append(row[0])
            count += 1
    with open(os.path.join(out_dir, "cover.txt"), "w") as f:
        f.write("\n".join(covers))
    with open(os.path.join(out_dir, "stego.txt"), "w") as f:
        f.write("\n".join(stegos))

if __name__ == '__main__':
    # extract_true_commonsense("train")
    # extract_true_commonsense("test")
    # extract_true_commonsense("dev")
    for dataset in ["movie","tweet","news"]:
        for alg in ["ac", "hc"]:
            csv2dir(csv_path="raw_data/{}/{}_0_1_{}.csv".format(dataset,dataset,alg), out_dir="dataset/{}/{}".format(dataset, alg))