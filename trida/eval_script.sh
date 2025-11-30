# Set the environment variables first before running the command.
export HF_ALLOW_CODE_EVAL=1
model_path=trillionlabs/Trida-7B-Preview


task=gsm8k
accelerate launch eval.py --tasks ${task} --batch_size 4 --num_fewshot 0 --limit 20 \
--confirm_run_unsafe_code --model fast_dllm_trida \
--model_args model_path=${model_path},threshold=0.9,show_speed=True,use_block_cache=False,add_bos_token=true,max_new_tokens=4000

task=hrm8k_gsm8k
accelerate launch eval.py --tasks ${task} --batch_size 4 \
--confirm_run_unsafe_code --model fast_dllm_trida --fewshot_as_multiturn --apply_chat_template \
--model_args model_path=${model_path},threshold=0.9,show_speed=True,add_bos_token=true,max_new_tokens=4000,use_block_cache=true

task=ifeval
accelerate launch eval.py --tasks ${task} --batch_size 32 \
--confirm_run_unsafe_code --model fast_dllm_trida --fewshot_as_multiturn --apply_chat_template \
--model_args model_path=${model_path},threshold=0.9,show_speed=True,use_block_cache=true

task=kobest
accelerate launch eval.py --tasks ${task} --batch_size 16 --num_fewshot 5 \
--confirm_run_unsafe_code --model fast_dllm_trida --fewshot_as_multiturn --apply_chat_template \
--model_args model_path=${model_path},add_bos_token=true



