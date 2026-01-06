echo 'TP8_warmup' | tee result.log
date >> result.log
python3 TP8_warmup.py
echo 'Warmup completed' | tee -a result.log
echo '--------------------------------' | tee -a result.log
echo 'TP8_latency' | tee result.log
date >> result.log
python3 TP8_latency.py | tee -a result.log
echo '--------------------------------' | tee -a result.log
echo 'TP8_throughput' | tee -a result.log
date >> result.log
python3 TP8_throughput.py | tee -a result.log
date >> result.log