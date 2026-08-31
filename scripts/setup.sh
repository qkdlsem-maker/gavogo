#!/usr/bin/env bash
mkdir -p data/raw/{highD,NGSIM,MiTra,ETRI} data/interim data/processed results/{models,tables,figures}
echo "데이터 배치:"
echo "  highD: data/raw/highD/  (NN_tracks.csv, NN_tracksMeta.csv)"
echo "  NGSIM: data/raw/NGSIM/  (trajectories-*.csv)"
echo "  MiTra: data/raw/MiTra/Data_T{n}/T{n}_DAll.csv"
echo "  ETRI : data/raw/ETRI/   (180906_s2d_l3_NN_..._lccrt.csv)"
