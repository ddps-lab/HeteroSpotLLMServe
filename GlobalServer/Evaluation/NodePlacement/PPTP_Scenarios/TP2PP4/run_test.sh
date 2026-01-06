echo 'TP2PP4_warmup' | tee result.log
date >> result.log
python3 TP2PP4_warmup.py
echo 'Warmup completed' | tee -a result.log
echo '--------------------------------' | tee -a result.log
echo 'TP2PP4_latency' | tee result.log
date >> result.log
python3 TP2PP4_latency.py | tee -a result.log
echo '--------------------------------' | tee -a result.log
echo 'TP2PP4_throughput' | tee -a result.log
date >> result.log
python3 TP2PP4_throughput.py | tee -a result.log
date >> result.log