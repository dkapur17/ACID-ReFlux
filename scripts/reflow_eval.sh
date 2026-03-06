# Script to evaluate 2-RF model with different solvers at with different sampling steps

mkdir -p eval_logs/reflow

#########################################################################################
# 1. n = 1

# Euler
echo "Euler n=1 1-RF"
modal run modal_app.py \
--action evaluate \
--checkpoint checkpoints/1-RF.pt \
--method cfm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 1 \
--solver euler \
--metrics kid \
--override | tee eval_logs/reflow/1rf_euler_1.txt


echo "Euler n=1 2-RF"
modal run modal_app.py \
--action evaluate \
--checkpoint checkpoints/2-RF.pt \
--method cfm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 1 \
--solver euler \
--metrics kid \
--override | tee eval_logs/reflow/2rf_euler_1.txt

echo "Euler n=1 2-RF-distilled"
modal run modal_app.py \
--action evaluate \
--checkpoint checkpoints/2-RF-distilled.pt \
--method cfm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 1 \
--solver euler \
--metrics kid \
--override | tee eval_logs/reflow/2rfd_euler_1.txt


#########################################################################################
# 2. n = 2

# Euler
echo "Euler n=2 1-RF"
modal run modal_app.py \
--action evaluate \
--checkpoint checkpoints/1-RF.pt \
--method cfm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 2 \
--solver euler \
--metrics kid \
--override | tee eval_logs/reflow/1rf_euler_2.txt


echo "Euler n=2 2-RF"
modal run modal_app.py \
--action evaluate \
--checkpoint checkpoints/2-RF.pt \
--method cfm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 2 \
--solver euler \
--metrics kid \
--override | tee eval_logs/reflow/2rf_euler_2.txt

echo "Euler n=2 2-RF-distilled"
modal run modal_app.py \
--action evaluate \
--checkpoint checkpoints/2-RF-distilled.pt \
--method cfm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 2 \
--solver euler \
--metrics kid \
--override | tee eval_logs/reflow/2rfd_euler_2.txt

#########################################################################################
# 3. n = 4

# Euler
echo "Euler n=4 1-RF"
modal run modal_app.py \
--action evaluate \
--checkpoint checkpoints/1-RF.pt \
--method cfm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 4 \
--solver euler \
--metrics kid \
--override | tee eval_logs/reflow/1rf_euler_4.txt


echo "Euler n=4 2-RF"
modal run modal_app.py \
--action evaluate \
--checkpoint checkpoints/2-RF.pt \
--method cfm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 4 \
--solver euler \
--metrics kid \
--override | tee eval_logs/reflow/2rf_euler_4.txt

echo "Euler n=4 2-RF-distilled"
modal run modal_app.py \
--action evaluate \
--checkpoint checkpoints/2-RF-distilled.pt \
--method cfm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 4 \
--solver euler \
--metrics kid \
--override | tee eval_logs/reflow/2rfd_euler_4.txt

#########################################################################################
# 4. n = 10

# Euler
echo "Euler n=10 1-RF"
modal run modal_app.py \
--action evaluate \
--checkpoint checkpoints/1-RF.pt \
--method cfm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 10 \
--solver euler \
--metrics kid \
--override | tee eval_logs/reflow/1rf_euler_10.txt


echo "Euler n=10 2-RF"
modal run modal_app.py \
--action evaluate \
--checkpoint checkpoints/2-RF.pt \
--method cfm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 10 \
--solver euler \
--metrics kid \
--override | tee eval_logs/reflow/2rf_euler_10.txt

echo "Euler n=10 2-RF-distilled"
modal run modal_app.py \
--action evaluate \
--checkpoint checkpoints/2-RF-distilled.pt \
--method cfm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 10 \
--solver euler \
--metrics kid \
--override | tee eval_logs/reflow/2rfd_euler_10.txt

#########################################################################################
# 5. n = 20

# Euler
echo "Euler n=20 1-RF"
modal run modal_app.py \
--action evaluate \
--checkpoint checkpoints/1-RF.pt \
--method cfm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 20 \
--solver euler \
--metrics kid \
--override | tee eval_logs/reflow/1rf_euler_20.txt


echo "Euler n=20 2-RF"
modal run modal_app.py \
--action evaluate \
--checkpoint checkpoints/2-RF.pt \
--method cfm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 20 \
--solver euler \
--metrics kid \
--override | tee eval_logs/reflow/2rf_euler_20.txt

echo "Euler n=20 2-RF-distilled"
modal run modal_app.py \
--action evaluate \
--checkpoint checkpoints/2-RF-distilled.pt \
--method cfm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 20 \
--solver euler \
--metrics kid \
--override | tee eval_logs/reflow/2rfd_euler_20.txt

#########################################################################################
# 6. n = 50

# Euler
echo "Euler n=50 1-RF"
modal run modal_app.py \
--action evaluate \
--checkpoint checkpoints/1-RF.pt \
--method cfm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 50 \
--solver euler \
--metrics kid \
--override | tee eval_logs/reflow/1rf_euler_50.txt


echo "Euler n=50 2-RF"
modal run modal_app.py \
--action evaluate \
--checkpoint checkpoints/2-RF.pt \
--method cfm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 50 \
--solver euler \
--metrics kid \
--override | tee eval_logs/reflow/2rf_euler_50.txt

echo "Euler n=50 2-RF-distilled"
modal run modal_app.py \
--action evaluate \
--checkpoint checkpoints/2-RF-distilled.pt \
--method cfm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 50 \
--solver euler \
--metrics kid \
--override | tee eval_logs/reflow/2rfd_euler_50.txt

#########################################################################################
# 7. n = 100

# Euler
echo "Euler n=100 1-RF"
modal run modal_app.py \
--action evaluate \
--checkpoint checkpoints/1-RF.pt \
--method cfm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 100 \
--solver euler \
--metrics kid \
--override | tee eval_logs/reflow/1rf_euler_100.txt


echo "Euler n=100 2-RF"
modal run modal_app.py \
--action evaluate \
--checkpoint checkpoints/2-RF.pt \
--method cfm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 100 \
--solver euler \
--metrics kid \
--override | tee eval_logs/reflow/2rf_euler_100.txt

echo "Euler n=100 2-RF-distilled"
modal run modal_app.py \
--action evaluate \
--checkpoint checkpoints/2-RF-distilled.pt \
--method cfm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 100 \
--solver euler \
--metrics kid \
--override | tee eval_logs/reflow/2rfd_euler_100.txt