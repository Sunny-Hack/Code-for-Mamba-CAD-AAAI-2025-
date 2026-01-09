#!/bin/bash
ps ux | grep -E 'python' | grep -v grep |awk '{print $2}' |xargs kill -s 9
python train.py --exp_name newDeepCAD -g 0 --data_root your_data_path
python test.py --exp_name newDeepCAD --mode rec --ckpt 4  -g 0 --data_root your_data_path
cd evaluation
python evaluate_ae_acc.py --src ../proj_log/newDeepCAD/results/test_4 
python evaluate_ae_cd.py --src ../proj_log/newDeepCAD/results/test_4 --parallel
cd ..