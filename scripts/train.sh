#!/usr/bin/env bash

set -o nounset                              # Treat unset variables as an error
set -e

echo "Container nvidia build = " $NVIDIA_BUILD_ID
nvidia-smi

src_dir=$1
model_ckpt=$2
model_config=$3
vocab_file=$4
train_file_basedir=$5
output_base_dir=$6

batch_size=${7:-1}
learning_rate=${8:-3e-5}
num_train_steps=${9:-43205}
max_seq_length=${10:-350}

num_accumulation_steps=${11:-1}
num_gpu=${12:-16}
precision=${13:-"fp32"}
use_xla=${14:-"true"}
optimizer_type=${15:-"adam"}
max_query_length=${16:-64}

do_lower_case=${17:-"true"}

# Document level QA parameters.
max_short_answers=${18:-10}
max_num_doc_feature=${19:-6}
filter_null_doc=${20:-"true"}

keep_topk=${21:-100}
single_pos_per_dupe=${22:-"true"}

global_loss=${23:-"doc_pos-hard_em"}
local_loss=${24:-"pos_loss"}

local_obj_alpha=${25:-1.0}
vat_reg_rate=${26:-4.0}
double_forward_reg_rate=${27:-0.0}
double_forward_loss=${28:-"hellinger"}
noise_epsilon=${29:-8.0}
noise_normalizer=${30:-"L2"}
merges_file=${31:-"None"}
vat_type=${32:-"global_local"}
teacher_temperature=${33:-1.0}

topk_for_train=${34:-100}

num_vat_est_iter=${35:-1}
accum_est=${36:-"false"}

layerwise_lr_decay=${37:-"-0.9"}

kl_alpha="1e-3"
kl_beta="1.0"

use_fp16=""
if [ "$precision" = "amp" ] ; then
  echo "fp16 activated!"
  use_fp16="--use_fp16=True"
elif [ "${precision}" = "fp16" ]; then
  echo "Using manual fp16"
  use_fp16="--manual_fp16=True"
fi

use_xla_tag=""
if [ "$use_xla" = "true" ] ; then
    echo "XLA activated"
    use_xla_tag="--use_xla"
fi

if [ "${do_lower_case}" = "true" ]; then
  case_flag="--do_lower_case=True"
else
  case_flag="--do_lower_case=False"
fi

mpi=""
use_hvd=""
if [ $num_gpu -gt 1 ] ; then
  mpi="mpirun -np $num_gpu -H localhost:$num_gpu \
  --allow-run-as-root -bind-to none -map-by slot \
  -x NCCL_DEBUG=INFO \
  -x LD_LIBRARY_PATH \
  -x PATH -mca pml ob1 -mca btl ^openib"
  use_hvd="--horovod"
fi

if [ $num_accumulation_steps -gt 1 ] ; then
    use_hvd="--horovod"
fi

rand_suffix=`shuf -i1-1000 -n1`
DATESTAMP=`date +'%y%m%d%H%M%S'`
output_dir="${output_base_dir}_${DATESTAMP}"

mkdir -p ${output_dir}

# For document-level QA, there are always negative passages.
v2_w_neg="true"

(
  echo "src_dir=${src_dir}"
  echo "vocab_file=${vocab_file}"
  echo "bert_config_file=${model_config}"
  echo "init_checkpoint=${model_ckpt}"
  echo "batch_size=${batch_size}"
  echo "learning_rate=${learning_rate}"
  echo "num_train_steps=${num_train_steps}"
  echo "max_seq_length=${max_seq_length}"
  echo "max_query_length=${max_query_length}"
  echo "vat_reg_rate=${vat_reg_rate}"
  echo "double_forward_reg_rate=${double_forward_reg_rate}"
  echo "double_forward_loss=${double_forward_loss}"
  echo "noise_epsilon=${noise_epsilon}"
  echo "single_pos_per_dupe=${single_pos_per_dupe}"
  echo "noise_normalizer=${noise_normalizer}"
  echo "kl_alpha=${kl_alpha}"
  echo "kl_beta=${kl_beta}"
  echo "max_num_doc_feature=${max_num_doc_feature}"
  echo "max_short_answers=${max_short_answers}"
  echo "filter_null_doc=${filter_null_doc}"
  echo "global_loss=${global_loss}"
  echo "topk_for_train=${topk_for_train}"
  echo "local_loss=${local_loss}"
  echo "local_obj_alpha=${local_obj_alpha}"
  echo "teacher_temperature=${teacher_temperature}"
  echo "vat_type=${vat_type}"
  echo "layerwise_lr_decay=${layerwise_lr_decay}"
)> ${output_dir}/exp.config

export PYTHONPATH="${src_dir}:$PYTHONPATH"


train_file_dir="${train_file_basedir}/topk_${keep_topk}_max-seq_${max_seq_length}_max-short-ans_${max_short_answers}_lower-case_${do_lower_case}"

if [ -d ${train_file_basedir} ]; then
  echo "Loading train tfrecords from ${train_file_dir}"
else
  echo "Can not load train tfrecords from ${train_file_dir}"
  exit 1
fi

echo "============================"
echo "Start training model"
echo "============================"

${mpi} python3 ${src_dir}/run_doc_qa_v2-1.py \
  --vocab_file=${vocab_file} \
  --bert_config_file=${model_config} \
  --init_checkpoint=${model_ckpt} \
  --do_train=True \
  --train_file_dir=${train_file_dir} \
  --do_predict=False \
  --debug=False \
  --train_batch_size=${batch_size} \
  --learning_rate=${learning_rate} \
  --num_train_steps=${num_train_steps} \
  --max_seq_length=${max_seq_length} \
  --max_query_length=${max_query_length} \
  --vat_reg_rate=${vat_reg_rate} \
  --double_forward_reg_rate=${double_forward_reg_rate} \
  --double_forward_loss=${double_forward_loss} \
  --noise_epsilon=${noise_epsilon} \
  --noise_normalizer=${noise_normalizer} \
  --layerwise_lr_decay=${layerwise_lr_decay} \
  --kl_alpha=${kl_alpha} \
  --kl_beta=${kl_beta} \
  --do_ema=False \
  --vat_type=${vat_type} \
  --single_pos_per_dupe=${single_pos_per_dupe} \
  --teacher_temperature=${teacher_temperature} \
  --max_num_doc_feature=${max_num_doc_feature} \
  --max_short_answers=${max_short_answers} \
  --filter_null_doc=${filter_null_doc} \
  --global_loss=${global_loss} \
  --local_loss=${local_loss} \
  --local_obj_alpha=${local_obj_alpha} \
  --topk_for_train=${topk_for_train} \
  --accum_est=${accum_est} \
  --num_vat_est_iter=${num_vat_est_iter} \
  ${case_flag} \
  --version_2_with_negative=${v2_w_neg} \
  --num_accumulation_steps=${num_accumulation_steps} \
  --optimizer_type=${optimizer_type} \
  ${use_hvd} ${use_xla_tag} ${use_fp16} \
  --output_dir=${output_dir} |& tee ${output_dir}/log.log
