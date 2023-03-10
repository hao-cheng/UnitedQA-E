#!/usr/bin/env bash
set -o nounset                              # Treat unset variables as an error
set -e


src_dir=$1
data_dir=$2
vocab_file=$3
do_lower_case=$4
output_base_dir=$5

num_split=${6:-16}
split=${7:-0}
max_seq_length=${8:-350}
max_query_length=${9:-64}

max_short_answers=${10:-10}
filter_null_doc=${11:-"true"}
keep_topk=${12:-100}

doc_stride=${13:-128}
use_doc_title=${14:-"true"}
pad_example=${15:-"false"}

if [ "${do_lower_case}" = "true" ]; then
  case_flag="--do_lower_case=True"
else
  case_flag="--do_lower_case=False"
fi

split_name="train"
json_file="${data_dir}/${split_name}-v2.0.json"

echo "Convert data into textline format"
output_dir="${output_base_dir}/topk_${keep_topk}_max-seq_${max_seq_length}_max-short-ans_${max_short_answers}_lower-case_${do_lower_case}"
mkdir -p ${output_dir}

python3 ${src_dir}/convert_example_to_textline.py \
  --vocab_file=${vocab_file} \
  --json_file=${json_file} \
  --split_name=${split_name} \
  --num_split=${num_split} \
  --split=${split} \
  --keep_topk=${keep_topk} \
  --max_seq_length=${max_seq_length} \
  --max_query_length=${max_query_length} \
  --max_short_answers=${max_short_answers} \
  --filter_null_doc=${filter_null_doc} \
  --use_doc_title=${use_doc_title} \
  --doc_stride=${doc_stride} \
  --pad_example=${pad_example} \
  ${case_flag} \
  --output_dir=${output_dir} |& tee ${output_dir}/log_${split_name}_${split}.log
