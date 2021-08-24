# 0.简介
多模态视频标签模型框架

# 1. 代码结构
- config---------------------# 模型选择，参数文件
- src------------------------# 数据加载/模型相关代码
- scripts--------------------# 数据预处理/训练/测试脚本
- checkpoints----------------# 模型权重/日志
- pretrained_models----------# 预训练模型
- dataset--------------------# 数据集和标签字典
- test_dataset---------------# 数据集和标签字典
- utils----------------------# 工具脚本
- results--------------------# 模型预测结果目录
- ReadMe.md

# 2. 环境配置

## 2.1 初始化环境
```bash
# 进入代码目录
cd MultiModal-Tagging
# 运行init.sh, 该脚本会创建代码的conda运行环境，并安装cuda10.1
./init.sh run
```
## 2.2 下载预训练模型
```bash
# 下载训练需要的预训练模型
./download_pretrain.sh
```
## 2.3 指定视频及特征目录
在特征方面我们只重新提取了视频的特征，其他文本，音频及图像特征均使用的是官方提供的
我们的数据路径默认在dataset目录下，请按照以下方式，创建软链接
```bash
ln -s /home/tione/notebook/algo-2021/dataset/tagging/tagging_dataset_train_5k/audio_npy/Vggish/tagging/ /home/tione/notebook/MultiModal-Tagging/dataset/audio_feat 
ln -s /home/tione/notebook/algo-2021/dataset/tagging/tagging_dataset_train_5k/image_jpg/tagging/ /home/tione/notebook/MultiModal-Tagging/dataset/img_feat
ln -s /home/tione/notebook/algo-2021/dataset/tagging/tagging_dataset_train_5k/text_txt/tagging/ /home/tione/notebook/MultiModal-Tagging/dataset/text_feat 
ln -s /home/tione/notebook/algo-2021/dataset/videos/video_5k/train_5k/ /home/tione/notebook/MultiModal-Tagging/dataset/videos 

```
## 2.4 下载或者重新提取视频特征
```bash
# 下载我们已经提取好的视频特征
cd dataset && wget https://taac1-1304126907.cos.ap-guangzhou.myqcloud.com/vit_B_16_384_features.tar.gz
tar -xvf vit_B_16_384_features.tar.gz
ln -s /home/tione/notebook/MultiModal-Tagging/dataset/vit_B_16_384_features /home/tione/notebook/MultiModal-Tagging/dataset/video_feat

# 重新提取
./extract_feature.sh "video_dir" "video_out_dir"
ln -s "video_out_dir" /home/tione/notebook/MultiModal-Tagging/dataset/video_feat
```

# 3. 训练流程
```bash
#直接执行脚本, 最后的模型文件在checkpoints/ 下面
./train.sh
```


# 4. 测试流程
## 4.1 指定数据特征目录
测试集的预测需要先提取特征，同样，我们只重新提取了视频特征，其余特征使用官方提供的, 默认测试集数据路径在test_dataset

```bash
# 指定特征对应的目录
ln -s test_videos_path /home/tione/notebook/MultiModal-Tagging/test_dataset/videos
ln -s test_videos_audio_feat /home/tione/notebook/MultiModal-Tagging/test_dataset/audio_npy
ln -s test_videos_img_feat /home/tione/notebook/MultiModal-Tagging/test_dataset/image_jpg
ln -s test_videos_txt_feat /home/tione/notebook/MultiModal-Tagging/test_dataset/text_txt

# 测试集的特征我们已经提取，可以直接下载
cd video_out_dir && wget https://taac1-1304126907.cos.ap-guangzhou.myqcloud.com/vit_B_16_384_features_test.tar.gz
# 或者重新提取
./extract_feature.sh "video_dir" "video_out_dir"

ln -s "video_out_dir" /home/tione/notebook/MultiModal-Tagging/test_dataset/video_feat

# 加载模型执行预测, result_path 可以不指定，默认在results.inference_result.json
./inference.sh "model_path" "result_path"
```

