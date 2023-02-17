# {U}nited{QA}: {A} Hybrid Approach for Open Domain Question Answering

This repository includes the codes and models for the extractive reader, UnitedQA-E, in our paper
[UnitedQA: A Hybrid Approach for Open Domain Question Answering](https://aclanthology.org/2021.acl-long.240).

If you find this useful, please cite the following paper:
```
@inproceedings{cheng-etal-2021-unitedqa,
    title = "{U}nited{QA}: {A} Hybrid Approach for Open Domain Question Answering",
    author = "Cheng, Hao  and
      Shen, Yelong  and
      Liu, Xiaodong  and
      He, Pengcheng  and
      Chen, Weizhu  and
      Gao, Jianfeng",
    booktitle = "Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers)",
    month = aug,
    year = "2021",
    address = "Online",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2021.acl-long.240",
    doi = "10.18653/v1/2021.acl-long.240",
    pages = "3080--3090",
}
```

## Requirements
* Python >= 3.6
* Tensorflow 1.14

The code in this repo has been tested with Tensorflow 1.14 on V100-32GB.
We highly recommend using the docker file for creating the enviroment.
All the following sample commands are based on using docker.

Build the docker image:
```
cd docker_file
docker build -t united_qa:v1.0 -f Dockerfile.cuda11 .
```

## Data Processing for the Extractive Reader
In our paper, we directly use the retrieval results from [DPR](https://github.com/facebookresearch/DPR). Please see their repository for downloading retriever outputs for NQ and TriviaQA. In the following, we use NQ as the walking example.

### Convert DPR retrieval results to reader format.
After downloading the retrieval json files and gold passage information (NQ only) into {retrieval_results_dir}, we first converts the training retrieval json into reader json by
```
python data_utils/convert_retriever_results_to_doc_qa.py \
  --input_dir="${retrieval_results_dir}" \
  --output_dir="${reader_data_dir}" \
  --splits="train" \
  --gold_passage_info_dir="${retrieval_results_dir}" \
  --dataset="nq" \
  --do_noise_match=true
```
Similar steps can be done for dev and test sets by changing the {splits} to "dev,test".

In the output {reader_data_dir}, there are five files
```
    .
    ├── train-v2.0.json             # Training file for reader.
    ├── dev-v2.0.json               # Dev file for reader.
    ├── test-v2.0.json              # Test file for reader.
    ├── nq_single_qid.dev.json      # Dictionary maps dev qid to question & answers.
    ├── nq_single_qid.test.json     # Dictionary maps test qid to question & answers.

```
Here, {split}-v2.0.json files are for reader training and the two additional dictionary files are used for evaluation purposes.


### Convert reader file into (serialized) TFRecord.
In order to train the model with Tensorflow, we have to do further processing steps to convert the original reader file into serialized data format.

For this step, we need to download the [pretrained electra model]() into ```{electra_model_dir}```.

*For training data, we can call the data-parallel script for processing
```
bash scripts/prepare_train_data.sh \
    ${src_code_dir} \
    ${reader_data_dir} \
    ${electra_model_dir}/vocab.txt \
    true \
    ${output_tfrecord_dir} \
    16 \
    0
```
Here, the script would then splits the original reader data into 16 shards and process the very first shard. The serialized data is then output to ${output_tfrecord_dir}.

*For eval data (dev/test), we need call another data-parallel script for processing
```
bash scripts/prepare_eval_data.sh \
    ${src_code_dir} \
    ${reader_data_dir} \
    ${electra_model_dir}/vocab.txt \
    true \
    ${output_tfrecord_dir} \
    dev \
    8 \
    0
```
Similar to training processing, the script would splits the dev reader data into multiple shards for parallel processing. Again, the serialized data is then output to ${output_tfrecord_dir}.

### Train the reader
The default training is going to use 16 V100-32GB GPUs. The training script can be called as
```
bash scripts/train.sh \
    ${src_code_dir} \
    ${electra_model_dir}/model.ckpt \
    ${electra_model_dir}/config.json \
    ${electra_model_dir}/vocab.txt \
    ${output_tfrecord_dir} \
    ${ckpt_output_dir} 
```
In particular, we have to adjust the ```num_train_steps``` based on the number of shards and number of GPUs.
For example, if we have ```N``` GPUs for training, we have to split the orginal data into ```k*N``` shards where ```k``` is a positive integer ```k>=1```.
This is required so that each GPU will take a unique portion of training data. Then, if each shard has ```M``` examples (questions) and we want to train the model for ```X``` epoches, then the total number of training steps is ```k*M*X```.

Also, ```layerwise_lr_decay``` of 0.9 / 0.8 is found to work reaonably well for large / base sized models.
For stability issues, we first train the original Electra model on SQuAD, and then fine-tune the model for ODQA.

### Evaluate the reader
Once the model is trained, we can then evaluate the model using the following script
```
bash scripts/eval.sh \
    ${src_code_dir} \
    ${ckpt_output_dir}/model_dir \
    ${electra_model_dir}/config.json \
    ${electra_model_dir}/vocab.txt \
    ${reader_data_dir} \
    ${output_tfrecord_dir} \
    ${ckpt_output_dir} 
```
Similar to training, the evaluation is also carried out in a data-parallel fashion. 
In the script, the evaluation is using 8 V100-32GB GPUs.
In other words, the dev set is split into 8 shards for parallel inference. Typically, it is better to have number of eval set shards smaller or equal than the number of GPUs for inference.

The final prediction is out at ${ckpt_output_dir}/dev|test where the ```test_predictions.json``` contains the best prediction for each question. 

We also release the model checkpoint for [UnitedQA-E](https://msrdeeplearning.blob.core.windows.net/udq-qa/unitedqa_e_model_weights.tgz) here.