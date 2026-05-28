set -e

# ========= Mode Selection =========
# am_asnorm: AM-Softmax + AS-Norm (recommended)
# circle:    CircleLoss baseline
mode="am_asnorm"

# ========= Common Config =========
encoder_name="conformer_cat" # conformer_cat | ecapa_tdnn_large | resnet34
embedding_dim=192

dataset="vox"
# num_classes=5994
num_classes=7205
num_blocks=6
train_csv_path="data/train.csv"

input_layer=conv2d2
pos_enc_layer_type=rel_pos # no_pos | rel_pos
trial_paths=(data/vox1_test.txt data/vox1_E_trials.txt data/vox1_H_trials.txt)
# trial_paths=(data/vox1_test.txt)
# ========= Resume Checkpoint =========
resume_ckpt_path="/hy-tmp/ldy/mfa_conformer/experiment_db_1/conv2d2/conformer_cat_6_192_amsoftmax_asnorm/epoch=8_cosine_eer=0.86.ckpt"
# Check if checkpoint exists
if [ -f "$resume_ckpt_path" ]; then
    echo "Found checkpoint: $resume_ckpt_path"
    echo "File size: $(du -h "$resume_ckpt_path" | cut -f1)"
    RESUME_ARG="--resume_ckpt_path $resume_ckpt_path"
else
    echo "Warning: checkpoint not found: $resume_ckpt_path"
    echo "Training from scratch"
    RESUME_ARG=""
fi

# ========= Training Hyperparameters (Common) =========
batch_size=180
num_workers=20
max_epochs=60
learning_rate=0.001
warmup_step=2000
step_size=4
gamma=0.5
weight_decay=0.0000001

# ========= Mode-Specific Config =========
extra_args=""
loss_name=""
mode_tag=""

if [ "$mode" = "am_asnorm" ]; then
  loss_name="amsoftmax"
  mode_tag="amsoftmax_asnorm"
  # AM-Softmax hyperparameters (grid-searchable)
  am_margin=0.30
  am_scale=40
  # AS-Norm post-processing (also prints asnorm_eer during evaluation)
  asnorm_top_n=200
  cohort_embedding_path="data/cohort_vox2_embeddings.npy"
  extra_args="--am_margin ${am_margin} --am_scale ${am_scale} --use_asnorm --asnorm_top_n ${asnorm_top_n} --cohort_embedding_path ${cohort_embedding_path}"
elif [ "$mode" = "circle" ]; then
  loss_name="circle"
  mode_tag="circle"
  circle_margin=0.25
  circle_scale=256
  extra_args="--circle_margin ${circle_margin} --circle_scale ${circle_scale}"
else
  echo "Unknown mode: ${mode}"
  echo "Valid modes: am_asnorm | circle"
  exit 1
fi

save_dir=experiment_EH_1/${input_layer}/${encoder_name}_${num_blocks}_${embedding_dim}_${mode_tag}

mkdir -p $save_dir
cp start.sh $save_dir
cp main.py $save_dir
cp -r module $save_dir
cp -r wenet $save_dir
cp -r scripts $save_dir
cp -r loss $save_dir
echo save_dir: $save_dir
echo resume_ckpt_path: $resume_ckpt_path

export CUDA_VISIBLE_DEVICES=0
python3 main.py \
        --batch_size $batch_size \
        --num_workers $num_workers \
        --max_epochs $max_epochs \
        --embedding_dim $embedding_dim \
        --save_dir $save_dir \
        --encoder_name ${encoder_name} \
        --train_csv_path $train_csv_path \
        --learning_rate $learning_rate \
        --warmup_step $warmup_step \
        --num_classes $num_classes \
        --trial_paths ${trial_paths[@]} \
        --loss_name $loss_name \
        --num_blocks $num_blocks \
        --step_size $step_size \
        --gamma $gamma \
        --weight_decay $weight_decay \
        --input_layer $input_layer \
        --pos_enc_layer_type $pos_enc_layer_type \
        # --seed 2073739378 \
        # --resume_ckpt_path $resume_ckpt_path
        $extra_args