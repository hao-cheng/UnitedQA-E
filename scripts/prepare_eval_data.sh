#!/usr/bin/env bash
set -o nounset                              # Treat unset variables as an error
set -e


src_dir=$1
data_dir=$2
vocab_file=$3
do_lower_case=$4

output_base_dir=$5

convert_split=${6:-"dev"}
num_split=${7:-8}
split=${8:-0}

max_seq_length=${9:-384}
max_query_length=${10:-64}

max_short_answers=${11:-10}
filter_null_doc=${12:-"false"}
keep_topk=${13:-100}

doc_stride=${14:-128}
use_doc_title=${15:-"true"}
pad_example=${16:-"false"}

if [ "${do_lower_case}" = "true" ]; then
  case_flag="--do_lower_case=True"
else
  case_flag="--do_lower_case=False"
fi

json_file="${data_dir}/${convert_split}-v2.0.json"

echo "Convert ${convert_split} data into textline format"
output_dir="${output_base_dir}/${convert_split}_topk_${keep_topk}_max-seq_${max_seq_length}_lower-case_${do_lower_case}_use-title_${use_doc_title}"
mkdir -p ${output_dir}

python3 ${src_dir}/convert_eval_example_to_textline.py \
  --vocab_file=${vocab_file} \
  --json_file=${json_file} \
  --split_name=${convert_split} \
  --num_split=${num_split} \
  --split=${split} \
  --keep_topk=${keep_topk} \
  --max_seq_length=${max_seq_length} \
  --max_query_length=${max_query_length} \
  --max_short_answers=${max_short_answers} \
  --filter_null_doc=false \
  --use_doc_title=${use_doc_title} \
  --doc_stride=${doc_stride} \
  --pad_example=false \
  ${case_flag} \
  --output_dir=${output_dir} |& tee ${output_dir}/log_${convert_split}_${split}.log
