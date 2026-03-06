# Script to evaluate 2-RF model with different solvers at with different sampling steps

mkdir -p eval_logs/reflux

#########################################################################################
# 1. n = 1

# Euler
echo "Euler n=1 ReFlux"
modal run modal_app.py \
--action evaluate \
--checkpoint checkpoints/reflux.pt \
--method cfm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 1 \
--solver euler \
--metrics kid \
--override | tee eval_logs/reflux/euler_1.txt


#########################################################################################
# 2. n = 2

# Euler
echo "Euler n=2 ReFlux"
modal run modal_app.py \
--action evaluate \
--checkpoint checkpoints/reflux.pt \
--method cfm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 2 \
--solver euler \
--metrics kid \
--override | tee eval_logs/reflux/euler_2.txt


#########################################################################################
# 3. n = 4

# Euler
echo "Euler n=4 ReFlux"
modal run modal_app.py \
--action evaluate \
--checkpoint checkpoints/reflux.pt \
--method cfm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 4 \
--solver euler \
--metrics kid \
--override | tee eval_logs/reflux/euler_4.txt


#########################################################################################
# 4. n = 10

# Euler
echo "Euler n=10 ReFlux"
modal run modal_app.py \
--action evaluate \
--checkpoint checkpoints/reflux.pt \
--method cfm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 10 \
--solver euler \
--metrics kid \
--override | tee eval_logs/reflux/euler_10.txt


#########################################################################################
# 5. n = 20

# Euler
echo "Euler n=20 ReFlux"
modal run modal_app.py \
--action evaluate \
--checkpoint checkpoints/reflux.pt \
--method cfm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 20 \
--solver euler \
--metrics kid \
--override | tee eval_logs/reflux/euler_20.txt


#########################################################################################
# 6. n = 50

# Euler
echo "Euler n=50 ReFlux"
modal run modal_app.py \
--action evaluate \
--checkpoint checkpoints/reflux.pt \
--method cfm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 50 \
--solver euler \
--metrics kid \
--override | tee eval_logs/reflux/euler_50.txt


#########################################################################################
# 7. n = 100

# Euler
echo "Euler n=100 ReFlux"
modal run modal_app.py \
--action evaluate \
--checkpoint checkpoints/reflux.pt \
--method cfm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 100 \
--solver euler \
--metrics kid \
--override | tee eval_logs/reflux/euler_100.txt