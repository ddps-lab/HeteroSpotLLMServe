echo 'test_warmup' | tee result.log
date >> result.log
python3 test_warmup.py
echo 'Warmup completed' | tee -a result.log
echo '--------------------------------' | tee -a result.log
echo 'test_time' | tee -a result.log
date >> result.log
python3 test_time.py | tee -a result.log
date >> result.log