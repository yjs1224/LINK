for dataset in aclImdb/ac
do
  python ground_concepts_simple.py --dataset ${dataset} --mode 1
  python find_neighbours.py --dataset ${dataset} --mode 1
  python filter_triple.py --dataset ${dataset} --mode 1
  done
