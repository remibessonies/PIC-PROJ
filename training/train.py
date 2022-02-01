pretrained_model_folder_input= sroie_folder_path / Path('layoutlm-base-uncased') # Define it so we can copy it into our working directory

pretrained_model_folder=Path('/output/working/layoutlm-base-uncased/') 
label_file=Path(dataset_directory, "labels.txt")

# Move to the script directory
os.chdir("/output/working/unilm/layoutlm/deprecated/examples/seq_labeling")

! cp -r "{pretrained_model_folder_input}" "{pretrained_model_folder}"
! sed -i 's/"num_attention_heads": 16,/"num_attention_heads": 12,/' "{pretrained_model_folder}/"config.json

! cat "/output/working/layoutlm-base-uncased/config.json"

! rm -rf /kaggle/working/dataset/cached*

! python run_seq_labeling.py \
                            --data_dir /kaggle/working/dataset \
                            --labels /kaggle/working/dataset/labels.txt \
                            --model_name_or_path "{pretrained_model_folder}" \
                            --model_type layoutlm \
                            --max_seq_length 512 \
                            --do_lower_case \
                            --do_train \
                            --num_train_epochs 10 \
                            --logging_steps 50 \
                            --save_steps -1 \
                            --output_dir output \
                            --overwrite_output_dir \
                            --per_gpu_train_batch_size 8 \
                            --per_gpu_eval_batch_size 16
