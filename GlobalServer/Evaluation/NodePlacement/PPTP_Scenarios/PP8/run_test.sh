echo 'PP8_warmup' | tee result.log
date >> result.log
python3 PP8_warmup.py
echo 'Warmup completed' | tee -a result.log
echo '--------------------------------' | tee -a result.log
echo 'PP8_latency' | tee result.log
date >> result.log
python3 PP8_latency.py | tee -a result.log
echo '--------------------------------' | tee -a result.log
echo 'PP8_throughput' | tee -a result.log
date >> result.log
python3 PP8_throughput.py | tee -a result.log
date >> result.log