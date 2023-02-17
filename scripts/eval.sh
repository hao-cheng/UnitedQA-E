#!/usr/bin/env bash
# This script contains sample run for SQUAD.

set -o nounset                              # Treat unset variables as an error
set -e

echo "Container nvidia build = " $NVIDIA_BUILD_ID

src_dir=$1
model_ckpt=$2
model_config=$3
vocab_file=$4
data_dir=$5
eval_file_basedir=$6
output_base_dir=$7

batch_size=${9:-24}
max_seq_length=${12:-350}

num_gpu=${14:-8}
max_query_length=${18:-64}

do_lower_case=${19:-"true"}

max_short_answers=${20:-10}

# Those are inference time variables.
use_rank=${23:-"false"}
sum=${24:-"true"}
decay=${25:-4.0}
use_doc_score=${27:-"true"}
keep_topk=${28:-100}
use_rel_score=${33:-"false"}

use_doc_title=${32:-"true"}

eval_split=${34:-"dev"}
max_answer_length=${36:-20}
infer_forward_k=${37:-0}

if [ "${do_lower_case}" = "true" ]; then
  case_flag="--do_lower_case=True"
else
  case_flag="--do_lower_case=False"
fi


output_dir="${output_base_dir}/${eval_split}"

mkdir -p ${output_dir}

# For document-level QA, there are always negative passages.
v2_w_neg="true"

(
  echo "src_dir=${src_dir}"
  echo "vocab_file=${vocab_file}"
  echo "bert_config_file=${model_config}"
  echo "init_checkpoint=${model_ckpt}"
  echo "batch_size=${batch_size}"
  echo "max_seq_length=${max_seq_length}"
  echo "max_query_length=${max_query_length}"
  echo "max_short_answers=${max_short_answers}"
  echo "use_doc_title=${use_doc_title}"
  echo "use_rank=${use_rank}"
  echo "max_answer_length=${max_answer_length}"
  echo "sum=${sum}"
  echo "decay=${decay}"
  echo "use_doc_score=${use_doc_score}"
  echo "infer_forward_k=${infer_forward_k}"
)> ${output_dir}/exp.config

export PYTHONPATH="${src_dir}:$PYTHONPATH"

test_file="${data_dir}/test-v2.0.json"
test_label_file="${data_dir}/nq_single_qid.test.json"
dev_file="${data_dir}/dev-v2.0.json"
dev_label_file="${data_dir}/nq_single_qid.dev.json"

if [ ${eval_split} == "dev" ]; then
  predict_file=${dev_file}
  label_file=${dev_label_file}
  
  echo "============================"
  echo "Evaluating the trained model on dev set"
  echo "============================"
  
  
  eval_file_dir="${eval_file_basedir}/dev_topk_${keep_topk}_max-seq_${max_seq_length}_lower-case_${do_lower_case}_use-title_${use_doc_title}"
else
  predict_file=${test_file}
  label_file=${test_label_file}
  
  echo "============================"
  echo "Evaluating the trained model on test set"
  echo "============================"
  
  eval_file_dir="${eval_file_basedir}/test_topk_${keep_topk}_max-seq_${max_seq_length}_lower-case_${do_lower_case}_use-title_${use_doc_title}"

fi

if [ -d ${eval_file_dir} ]; then
  echo "Reading eval files from ${eval_file_dir}"
else
  mkdir -p ${eval_file_dir}
fi

if [ -f ${eval_file_dir}/eval.tf_record-0 ]; then
  num_eval_split=$(ls ${eval_file_dir}/eval.tf_record-* | wc -l)
else
  num_eval_split=${num_gpu}
fi

if [ ${num_gpu} -lt ${num_eval_split} ]; then
  echo "There are ${num_eval_split} splits for evaluation"
  echo "Only using ${num_gpu} for parallel evaluations"
  exit 1
fi

if [ -f "${output_dir}/nbest_predictions-0.json" ]; then
  rm ${output_dir}/nbest_predictions-*.json
fi

if [ ${num_eval_split} = "1" ]; then
  i=0
  echo "Launch eval $i"
    python3 ${src_dir}/run_doc_qa_v3.py \
      --vocab_file=${vocab_file} \
      --bert_config_file=${model_config} \
      --init_checkpoint=${model_ckpt} \
      --tfrecord_dir=${eval_file_dir} \
      --do_train=False \
      --do_predict=True \
      --debug=False \
      --predict_file=${predict_file} \
      --train_batch_size=${batch_size} \
      --predict_batch_size=${batch_size} \
      --num_eval_split=${num_eval_split} \
      --eval_split_id=${i} \
      --max_seq_length=${max_seq_length} \
      --max_query_length=${max_query_length} \
      --max_short_answers=${max_short_answers} \
      --max_answer_length=${max_answer_length} \
      --filter_null_doc=False \
      --doc_stride=128 \
      --infer_forward_k=${infer_forward_k} \
      --topk_for_infer=${keep_topk} \
      --use_doc_title=${use_doc_title} \
      ${case_flag} \
      --version_2_with_negative=${v2_w_neg} \
      --output_dir=${output_dir} |& tee ${output_dir}/eval_log_${i}.log 

else
# For multiple split evaluations, launches all in parallel.
ub=$(expr ${num_eval_split} - 1)
for ((i=0; i<$num_eval_split; i++));
do
  echo "Launch eval $i"
  if [ $i -lt ${ub} ]; then
    python3 ${src_dir}/run_doc_qa_v3.py \
      --vocab_file=${vocab_file} \
      --bert_config_file=${model_config} \
      --init_checkpoint=${model_ckpt} \
      --tfrecord_dir=${eval_file_dir} \
      --do_train=False \
      --do_predict=True \
      --debug=False \
      --predict_file=${predict_file} \
      --train_batch_size=${batch_size} \
      --predict_batch_size=${batch_size} \
      --num_eval_split=${num_eval_split} \
      --eval_split_id=${i} \
      --max_seq_length=${max_seq_length} \
      --max_query_length=${max_query_length} \
      --max_short_answers=${max_short_answers} \
      --max_answer_length=${max_answer_length} \
      --filter_null_doc=False \
      --doc_stride=128 \
      --infer_forward_k=${infer_forward_k} \
      --topk_for_infer=${keep_topk} \
      --use_doc_title=${use_doc_title} \
      ${case_flag} \
      --version_2_with_negative=${v2_w_neg} \
      --output_dir=${output_dir} |& tee ${output_dir}/eval_log_${i}.log &
  else
    python3 ${src_dir}/run_doc_qa_v3.py \
      --vocab_file=${vocab_file} \
      --bert_config_file=${model_config} \
      --init_checkpoint=${model_ckpt} \
      --tfrecord_dir=${eval_file_dir} \
      --do_train=False \
      --do_predict=True \
      --debug=False \
      --predict_file=${predict_file} \
      --train_batch_size=${batch_size} \
      --predict_batch_size=${batch_size} \
      --num_eval_split=${num_eval_split} \
      --eval_split_id=${i} \
      --max_seq_length=${max_seq_length} \
      --max_query_length=${max_query_length} \
      --max_short_answers=${max_short_answers} \
      --max_answer_length=${max_answer_length} \
      --filter_null_doc=False \
      --doc_stride=128 \
      --infer_forward_k=${infer_forward_k} \
      --topk_for_infer=${keep_topk} \
      --use_doc_title=${use_doc_title} \
      ${case_flag} \
      --version_2_with_negative=${v2_w_neg} \
      --output_dir=${output_dir} |& tee ${output_dir}/eval_log_${i}.log
  fi
done
fi

num_file=$(ls ${output_dir}/nbest_predictions-*.json | wc -l)

while [ ${num_file} -lt ${num_eval_split} ];
do
  echo "Waiting for the prediction to be done"
  echo "${num_file} out of ${num_eval_split} is done"
  sleep 10m
  num_file=$(ls ${output_dir}/nbest_predictions-*.json | wc -l)
done

jq -s add ${output_dir}/nbest_predictions-*.json > ${output_dir}/nbest_predictions.json

# Extracts answers from n-best.
python3 ${src_dir}/qa_utils/extract_answers.py \
  --nbest_file=${output_dir}/nbest_predictions.json \
  --predictions_file=${output_dir}/${eval_split}_predictions.json \
  --use_rank=${use_rank} \
  --use_rel_score=${use_rel_score} \
  --sum=${sum} \
  --decay=${decay} \
  --max_para=${keep_topk} \
  --use_doc_score=${use_doc_score}

python3 ${src_dir}/qa_utils/od_qa_eval.py \
  --dataset_file ${label_file} \
  --prediction_file ${output_dir}/${eval_split}_predictions.json \
  --out_file ${output_dir}/${eval_split}.metrics
