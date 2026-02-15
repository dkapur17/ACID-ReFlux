# Script to evaluate 2-RF model with different solvers at with different sampling steps

mkdir -p eval_logs/reflow

#########################################################################################
# 1. n = 1

# Euler
echo "Euler n=1"
modal run modal_app.py \
--action evaluate \
--checkpoint logs/reflow/reflow_20260214_160857/2-RF/checkpoints/2-RF_final.pt \
--method cfm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 1 \
--solver euler \
--metrics kid \
--override | tee eval_logs/reflow/euler_1.txt

# Heun
echo "Heun n=1"
modal run modal_app.py \
--action evaluate \
--checkpoint logs/reflow/reflow_20260214_160857/2-RF/checkpoints/2-RF_final.pt \
--method cfm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 1 \
--solver heun \
--metrics kid \
--override | tee eval_logs/reflow/heun_1.txt

# RK2
echo "RK2 n=1"
modal run modal_app.py \
--action evaluate \
--checkpoint logs/reflow/reflow_20260214_160857/2-RF/checkpoints/2-RF_final.pt \
--method cfm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 1 \
--solver rk2 \
--metrics kid \
--override | tee eval_logs/reflow/rk2_1.txt

# RK4
echo "RK4 n=1"
modal run modal_app.py \
--action evaluate \
--checkpoint logs/reflow/reflow_20260214_160857/2-RF/checkpoints/2-RF_final.pt \
--method cfm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 1 \
--solver rk4 \
--metrics kid \
--override | tee eval_logs/reflow/rk4_1.txt


#########################################################################################
# 2. n = 2

# Euler
echo "Euler n=2"
modal run modal_app.py \
--action evaluate \
--checkpoint logs/reflow/reflow_20260214_160857/2-RF/checkpoints/2-RF_final.pt \
--method cfm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 2 \
--solver euler \
--metrics kid \
--override | tee eval_logs/reflow/euler_2.txt

# Heun
echo "Heun n=2"
modal run modal_app.py \
--action evaluate \
--checkpoint logs/reflow/reflow_20260214_160857/2-RF/checkpoints/2-RF_final.pt \
--method cfm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 2 \
--solver heun \
--metrics kid \
--override | tee eval_logs/reflow/heun_2.txt

# RK2
echo "RK2 n=2"
modal run modal_app.py \
--action evaluate \
--checkpoint logs/reflow/reflow_20260214_160857/2-RF/checkpoints/2-RF_final.pt \
--method cfm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 2 \
--solver rk2 \
--metrics kid \
--override | tee eval_logs/reflow/rk2_2.txt

# RK4
echo "RK4 n=2"
modal run modal_app.py \
--action evaluate \
--checkpoint logs/reflow/reflow_20260214_160857/2-RF/checkpoints/2-RF_final.pt \
--method cfm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 2 \
--solver rk4 \
--metrics kid \
--override | tee eval_logs/reflow/rk4_2.txt


#########################################################################################
# 3. n = 4

# Euler
echo "Euler n=4"
modal run modal_app.py \
--action evaluate \
--checkpoint logs/reflow/reflow_20260214_160857/2-RF/checkpoints/2-RF_final.pt \
--method cfm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 4 \
--solver euler \
--metrics kid \
--override | tee eval_logs/reflow/euler_4.txt

# Heun
echo "Heun n=4"
modal run modal_app.py \
--action evaluate \
--checkpoint logs/reflow/reflow_20260214_160857/2-RF/checkpoints/2-RF_final.pt \
--method cfm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 4 \
--solver heun \
--metrics kid \
--override | tee eval_logs/reflow/heun_4.txt

# RK2
echo "RK2 n=4"
modal run modal_app.py \
--action evaluate \
--checkpoint logs/reflow/reflow_20260214_160857/2-RF/checkpoints/2-RF_final.pt \
--method cfm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 4 \
--solver rk2 \
--metrics kid \
--override | tee eval_logs/reflow/rk2_4.txt

# RK4
echo "RK4 n=4"
modal run modal_app.py \
--action evaluate \
--checkpoint logs/reflow/reflow_20260214_160857/2-RF/checkpoints/2-RF_final.pt \
--method cfm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 4 \
--solver rk4 \
--metrics kid \
--override | tee eval_logs/reflow/rk4_4.txt

#########################################################################################
# 4. n = 10

# Euler
echo "Euler n=10"
modal run modal_app.py \
--action evaluate \
--checkpoint logs/reflow/reflow_20260214_160857/2-RF/checkpoints/2-RF_final.pt \
--method cfm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 10 \
--solver euler \
--metrics kid \
--override | tee eval_logs/reflow/euler_10.txt

# Heun
echo "Heun n=10"
modal run modal_app.py \
--action evaluate \
--checkpoint logs/reflow/reflow_20260214_160857/2-RF/checkpoints/2-RF_final.pt \
--method cfm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 10 \
--solver heun \
--metrics kid \
--override | tee eval_logs/reflow/heun_10.txt

# RK2
echo "RK2 n=10"
modal run modal_app.py \
--action evaluate \
--checkpoint logs/reflow/reflow_20260214_160857/2-RF/checkpoints/2-RF_final.pt \
--method cfm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 10 \
--solver rk2 \
--metrics kid \
--override | tee eval_logs/reflow/rk2_10.txt

# RK4
echo "RK4 n=10"
modal run modal_app.py \
--action evaluate \
--checkpoint logs/reflow/reflow_20260214_160857/2-RF/checkpoints/2-RF_final.pt \
--method cfm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 10 \
--solver rk4 \
--metrics kid \
--override | tee eval_logs/reflow/rk4_10.txt


#########################################################################################
# 5. n = 20

# Euler
echo "Euler n=20"
modal run modal_app.py \
--action evaluate \
--checkpoint logs/reflow/reflow_20260214_160857/2-RF/checkpoints/2-RF_final.pt \
--method cfm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 20 \
--solver euler \
--metrics kid \
--override | tee eval_logs/reflow/euler_20.txt

# Heun
echo "Heun n=20"
modal run modal_app.py \
--action evaluate \
--checkpoint logs/reflow/reflow_20260214_160857/2-RF/checkpoints/2-RF_final.pt \
--method cfm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 20 \
--solver heun \
--metrics kid \
--override | tee eval_logs/reflow/heun_20.txt

# RK2
echo "RK2 n=20"
modal run modal_app.py \
--action evaluate \
--checkpoint logs/reflow/reflow_20260214_160857/2-RF/checkpoints/2-RF_final.pt \
--method cfm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 20 \
--solver rk2 \
--metrics kid \
--override | tee eval_logs/reflow/rk2_20.txt

# RK4
echo "RK4 n=20"
modal run modal_app.py \
--action evaluate \
--checkpoint logs/reflow/reflow_20260214_160857/2-RF/checkpoints/2-RF_final.pt \
--method cfm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 20 \
--solver rk4 \
--metrics kid \
--override | tee eval_logs/reflow/rk4_20.txt


#########################################################################################
# 6. n = 50

# Euler
echo "Euler n=50"
modal run modal_app.py \
--action evaluate \
--checkpoint logs/reflow/reflow_20260214_160857/2-RF/checkpoints/2-RF_final.pt \
--method cfm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 50 \
--solver euler \
--metrics kid \
--override | tee eval_logs/reflow/euler_50.txt

# Heun
echo "Heun n=50"
modal run modal_app.py \
--action evaluate \
--checkpoint logs/reflow/reflow_20260214_160857/2-RF/checkpoints/2-RF_final.pt \
--method cfm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 50 \
--solver heun \
--metrics kid \
--override | tee eval_logs/reflow/heun_50.txt

# RK2
echo "RK2 n=50"
modal run modal_app.py \
--action evaluate \
--checkpoint logs/reflow/reflow_20260214_160857/2-RF/checkpoints/2-RF_final.pt \
--method cfm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 50 \
--solver rk2 \
--metrics kid \
--override | tee eval_logs/reflow/rk2_50.txt

# RK4
echo "RK4 n=50"
modal run modal_app.py \
--action evaluate \
--checkpoint logs/reflow/reflow_20260214_160857/2-RF/checkpoints/2-RF_final.pt \
--method cfm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 50 \
--solver rk4 \
--metrics kid \
--override | tee eval_logs/reflow/rk4_50.txt


#########################################################################################
# 7. n = 100

# Euler
echo "Euler n=100"
modal run modal_app.py \
--action evaluate \
--checkpoint logs/reflow/reflow_20260214_160857/2-RF/checkpoints/2-RF_final.pt \
--method cfm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 100 \
--solver euler \
--metrics kid \
--override | tee eval_logs/reflow/euler_100.txt

# Heun
echo "Heun n=100"
modal run modal_app.py \
--action evaluate \
--checkpoint logs/reflow/reflow_20260214_160857/2-RF/checkpoints/2-RF_final.pt \
--method cfm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 100 \
--solver heun \
--metrics kid \
--override | tee eval_logs/reflow/heun_100.txt

# RK2
echo "RK2 n=100"
modal run modal_app.py \
--action evaluate \
--checkpoint logs/reflow/reflow_20260214_160857/2-RF/checkpoints/2-RF_final.pt \
--method cfm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 100 \
--solver rk2 \
--metrics kid \
--override | tee eval_logs/reflow/rk2_100.txt

# RK4
echo "RK4 n=100"
modal run modal_app.py \
--action evaluate \
--checkpoint logs/reflow/reflow_20260214_160857/2-RF/checkpoints/2-RF_final.pt \
--method cfm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 100 \
--solver rk4 \
--metrics kid \
--override | tee eval_logs/reflow/rk4_100.txt

