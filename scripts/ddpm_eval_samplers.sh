mkdir -p eval_logs/ddpm

echo "DDPM n=1"
modal run modal_app.py \
--action evaluate \
--checkpoint logs/ddpm_37m/ddpm_20260214_032141/checkpoints/ddpm_final.pt \
--method ddpm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 1 \
--sampler ddpm \
--metrics kid \
--override | tee eval_logs/ddpm/ddpm_1.txt

echo "DDIM n=1"
modal run modal_app.py \
--action evaluate \
--checkpoint logs/ddpm_37m/ddpm_20260214_032141/checkpoints/ddpm_final.pt \
--method ddpm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 1 \
--sampler ddim \
--metrics kid \
--override | tee eval_logs/ddpm/ddim_1.txt

echo "DDPM n=2"
modal run modal_app.py \
--action evaluate \
--checkpoint logs/ddpm_37m/ddpm_20260214_032141/checkpoints/ddpm_final.pt \
--method ddpm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 2 \
--sampler ddpm \
--metrics kid \
--override | tee eval_logs/ddpm/ddpm_2.txt

echo "DDIM n=2"
modal run modal_app.py \
--action evaluate \
--checkpoint logs/ddpm_37m/ddpm_20260214_032141/checkpoints/ddpm_final.pt \
--method ddpm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 2 \
--sampler ddim \
--metrics kid \
--override | tee eval_logs/ddpm/ddim_2.txt

echo "DDPM n=4"
modal run modal_app.py \
--action evaluate \
--checkpoint logs/ddpm_37m/ddpm_20260214_032141/checkpoints/ddpm_final.pt \
--method ddpm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 4 \
--sampler ddpm \
--metrics kid \
--override | tee eval_logs/ddpm/ddpm_4.txt

echo "DDIM n=4"
modal run modal_app.py \
--action evaluate \
--checkpoint logs/ddpm_37m/ddpm_20260214_032141/checkpoints/ddpm_final.pt \
--method ddpm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 4 \
--sampler ddim \
--metrics kid \
--override | tee eval_logs/ddpm/ddim_4.txt

echo "DDPM n=10"
modal run modal_app.py \
--action evaluate \
--checkpoint logs/ddpm_37m/ddpm_20260214_032141/checkpoints/ddpm_final.pt \
--method ddpm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 10 \
--sampler ddpm \
--metrics kid \
--override | tee eval_logs/ddpm/ddpm_10.txt

echo "DDIM n=10"
modal run modal_app.py \
--action evaluate \
--checkpoint logs/ddpm_37m/ddpm_20260214_032141/checkpoints/ddpm_final.pt \
--method ddpm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 10 \
--sampler ddim \
--metrics kid \
--override | tee eval_logs/ddpm/ddim_10.txt

echo "DDPM n=20"
modal run modal_app.py \
--action evaluate \
--checkpoint logs/ddpm_37m/ddpm_20260214_032141/checkpoints/ddpm_final.pt \
--method ddpm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 20 \
--sampler ddpm \
--metrics kid \
--override | tee eval_logs/ddpm/ddpm_20.txt

echo "DDIM n=20"
modal run modal_app.py \
--action evaluate \
--checkpoint logs/ddpm_37m/ddpm_20260214_032141/checkpoints/ddpm_final.pt \
--method ddpm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 20 \
--sampler ddim \
--metrics kid \
--override | tee eval_logs/ddpm/ddim_20.txt

echo "DDPM n=50"
modal run modal_app.py \
--action evaluate \
--checkpoint logs/ddpm_37m/ddpm_20260214_032141/checkpoints/ddpm_final.pt \
--method ddpm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 50 \
--sampler ddpm \
--metrics kid \
--override | tee eval_logs/ddpm/ddpm_50.txt

echo "DDIM n=50"
modal run modal_app.py \
--action evaluate \
--checkpoint logs/ddpm_37m/ddpm_20260214_032141/checkpoints/ddpm_final.pt \
--method ddpm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 50 \
--sampler ddim \
--metrics kid \
--override | tee eval_logs/ddpm/ddim_50.txt

echo "DDPM n=100"
modal run modal_app.py \
--action evaluate \
--checkpoint logs/ddpm_37m/ddpm_20260214_032141/checkpoints/ddpm_final.pt \
--method ddpm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 100 \
--sampler ddpm \
--metrics kid \
--override | tee eval_logs/ddpm/ddpm_100.txt

echo "DDIM n=100"
modal run modal_app.py \
--action evaluate \
--checkpoint logs/ddpm_37m/ddpm_20260214_032141/checkpoints/ddpm_final.pt \
--method ddpm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 100 \
--sampler ddim \
--metrics kid \
--override | tee eval_logs/ddpm/ddim_100.txt

echo "DDPM n=200"
modal run modal_app.py \
--action evaluate \
--checkpoint logs/ddpm_37m/ddpm_20260214_032141/checkpoints/ddpm_final.pt \
--method ddpm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 200 \
--sampler ddpm \
--metrics kid \
--override | tee eval_logs/ddpm/ddpm_200.txt

echo "DDIM n=200"
modal run modal_app.py \
--action evaluate \
--checkpoint logs/ddpm_37m/ddpm_20260214_032141/checkpoints/ddpm_final.pt \
--method ddpm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 200 \
--sampler ddim \
--metrics kid \
--override | tee eval_logs/ddpm/ddim_200.txt

echo "DDPM n=500"
modal run modal_app.py \
--action evaluate \
--checkpoint logs/ddpm_37m/ddpm_20260214_032141/checkpoints/ddpm_final.pt \
--method ddpm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 500 \
--sampler ddpm \
--metrics kid \
--override | tee eval_logs/ddpm/ddpm_500.txt

echo "DDIM n=500"
modal run modal_app.py \
--action evaluate \
--checkpoint logs/ddpm_37m/ddpm_20260214_032141/checkpoints/ddpm_final.pt \
--method ddpm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 500 \
--sampler ddim \
--metrics kid \
--override | tee eval_logs/ddpm/ddim_500.txt

echo "DDPM n=1000"
modal run modal_app.py \
--action evaluate \
--checkpoint logs/ddpm_37m/ddpm_20260214_032141/checkpoints/ddpm_final.pt \
--method ddpm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 1000 \
--sampler ddpm \
--metrics kid \
--override | tee eval_logs/ddpm/ddpm_1000.txt

echo "DDIM n=1000"
modal run modal_app.py \
--action evaluate \
--checkpoint logs/ddpm_37m/ddpm_20260214_032141/checkpoints/ddpm_final.pt \
--method ddpm \
--batch-size 128 \
--num-samples 1000 \
--num-steps 1000 \
--sampler ddim \
--metrics kid \
--override | tee eval_logs/ddpm/ddim_1000.txt

