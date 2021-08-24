#encoding: utf-8
import sys,os
sys.path.append(os.getcwd())
import glob
import tensorflow.compat.v1 as tf
import numpy as np
import cv2
import argparse
import time
import traceback
import json
import utils.tokenization as tokenization
from utils.train_util import get_label_name_dict
from src.feats_extract.multimodal_feature_extract import MultiModalFeatureExtract

os.environ["CUDA_VISIBLE_DEVICES"]='0'

#################Inference Utils#################
tokokenizer = tokenization.FullTokenizer(vocab_file='pretrained/bert/chinese_L-12_H-768_A-12/vocab.txt')
class TaggingModel():
    def __init__(self, configs):
        tag_id_file = configs.get('tag_id_file', None)
        model_pb = configs.get('model_pb', None)
        extract_text = configs.get('extract_text', False)
        if tag_id_file is None:
            raise
        else:
            self.label_name_dict = get_label_name_dict(tag_id_file, None)
        if model_pb is None:
            raise
        else:
            config = tf.ConfigProto(allow_soft_placement=True)
            config.gpu_options.allow_growth = True
            self.sess = tf.Session(config=config)
            meta_graph_def = tf.saved_model.loader.load(self.sess, [tf.saved_model.tag_constants.SERVING], model_pb)
            signature_def = meta_graph_def.signature_def
            self.signature = signature_def[tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

        batch_size = configs.get('video_feats_extractor_batch_size', 8)
        imgfeat_extractor = configs.get('imgfeat_extractor', 'Youtube8M')
        self.feat_extractor = MultiModalFeatureExtract(batch_size=batch_size, imgfeat_extractor= imgfeat_extractor, 
                             extract_video = True, extract_audio = True, extract_text = extract_text)

    def image_preprocess(self, image, rescale=224):
        #resize to 224 and normlize to 0-1, then perform f(x)= 2*(x-0.5)
        if isinstance(image, type(None)):
          print("WARNING: test input image is None")
          return np.zeros((rescale, rescale, 3))
        if image.shape[0] !=rescale:
          image = cv2.resize(image, (rescale, rescale))
        image = 2*(image/(np.max(image)+1e-10) - 0.5)
        return image

    def text_preprocess(self, txt,max_len=128):
        tokens = ['[CLS]'] + tokokenizer.tokenize(txt)
        ids = tokokenizer.convert_tokens_to_ids(tokens)
        ids = ids[:max_len]
        ids = ids + [0]*(max_len-len(ids))
        return ids


    def preprocess(self, feat_dict, max_frame_num=300):
        ret_dict = {}
        for feat_type in feat_dict:
            if feat_type=='video':
                feats = np.zeros((max_frame_num, len(feat_dict['video'][0])))
                valid_num = min(max_frame_num, len(feat_dict['video']))
                feats[:valid_num] = feat_dict['video']
            elif feat_type=='audio':
                feats = np.zeros((max_frame_num, len(feat_dict['audio'][0])))
                valid_num = min(max_frame_num, len(feat_dict['audio']))
                feats[:valid_num] = feat_dict['audio']
            elif feat_type=='text':
                feats = self.text_preprocess(feat_dict['text'], 128)
            elif feat_type == 'image':
                feats = self.image_preprocess(feat_dict['image'])
            else:
                raise
            ret_dict[feat_type] = feats
        return ret_dict


    def inference(self, test_file):
        with self.sess.as_default() as sess:
            start_time = time.time()
            feat_dict = self.feat_extractor.extract_feat(test_file, save=False)
            end_time = time.time()
            print("feature extract cost time: {} sec".format(end_time -start_time))
            if 'text' in feat_dict:
                print(feat_dict['text'])
            else:
                feat_dict['text'] = ""

            feat_dict_preprocess = self.preprocess(feat_dict)
            feed_dict ={}
            
            # Get input tensor.
            for key in feat_dict:
                if key in self.signature.inputs:
                  feed_dict[self.signature.inputs[key].name] = [feat_dict_preprocess[key]]
                
            if 'video_frames_num' in self.signature.inputs:
                feed_dict[self.signature.inputs['video_frames_num'].name] = [len(feat_dict['video'])]
            if 'audio_frames_num' in self.signature.inputs:
                feed_dict[self.signature.inputs['audio_frames_num'].name] = [len(feat_dict['audio'])]
                
            # Get output tensor.
            class_indexes = self.signature.outputs['class_indexes'].name
            predictions = self.signature.outputs['predictions'].name
            #video_embedding = self.signature.outputs['video_embedding'].name #(Optional)
            
            start_time = time.time()
            class_indexes,predictions = sess.run([class_indexes,predictions], feed_dict)
            end_time = time.time()
            
            print("multi-modal tagging model forward cost time: {} sec".format(end_time -start_time))


            labels=[self.label_name_dict[index] for index in class_indexes[0]]
            scores = predictions[0]

        return labels, scores

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_pb', default='checkpoints/ds/export/step_10000_0.6879',type=str)
    parser.add_argument('--tag_id_file', default='../dataset/label_id.txt')
    parser.add_argument('--test_dir', default='../SceneSeg/data/train799/scene_video')
    parser.add_argument('--postfix', default='.mp4', type=str, help='test file type')
    parser.add_argument('--extract_text', type=int, default=0)
    parser.add_argument('--top_k', type=int, default=20)
    parser.add_argument('--output', default="results/result_for_vis.txt", type=str) #用于可视化文件
    parser.add_argument('--output_json', default="results/outjson.txt", type=str) #用于模型精度评估
    args = parser.parse_args()
    
    configs={'tag_id_file': args.tag_id_file, 'model_pb': args.model_pb, 'extract_text': args.extract_text}
    model = TaggingModel(configs)
    test_files = glob.glob(args.test_dir+'/*'+args.postfix)
    test_files.sort()    
    output_result = {}
    
    #clean temp file
    if os.path.exists(args.output):
        os.remove(args.output)

    for test_file in test_files:
        print(test_file)
        try:
          labels, scores = model.inference(test_file)
        except:
          print(traceback.format_exc())
        # print(test_file.split("/")[-1].split(".m")[0].split("#"))
        video_id, segment_id, start_time, end_time, _ = test_file.split("/")[-1].split(".m")[0].split("#")        
        # print(video_id, segment_id, start_time, end_time)

        if args.output_json is not None:
            cur_output = {"segment": [start_time, end_time], "labels": labels[:args.top_k], "scores": ["%.2f" % scores[i] for i in range(args.top_k)]}
            video_id = video_id + '.mp4'
            if video_id not in output_result:
                output_result[video_id] = {"result": [cur_output]}
            else:
                output_result[video_id]["result"].append(cur_output)

        if args.output is not None:
            with open(args.output, 'a+') as f:
                video_id = test_file.split('/')[-1].split('.m')[0]
                scores = [scores[i] for i in range(args.top_k)]
                f.write("{}\t{}\n".format(video_id, "\t".join(["{}##{:.3f}".format(labels[i], scores[i]) for i in range(len(scores))])))
        print("-"*100)
    
    for key in output_result:
        output_result[key]["result"].sort(key=lambda x: float(x["segment"][0]))    

    with open(args.output_json, 'w', encoding="utf-8") as f:
        json.dump(output_result, f, ensure_ascii=False, indent = 4)
