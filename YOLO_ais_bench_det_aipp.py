import argparse
import time 
import cv2
import numpy as np
import os

from ais_bench.infer.interface import InferSession


class YOLO:
    """YOLO object detection model class for handling inference"""

    def __init__(self, om_model, imgsz=(640, 640), device_id=0, model_ndtype=np.single, mode="static", postprocess_type="v8", aipp=False):
        """
        Initialization.

        Args:
            om_model (str): Path to the om model.
        """
        
        # 构建ais_bench推理引擎
        self.session = InferSession(device_id=device_id, model_path=om_model)
        
        # Numpy dtype: support both FP32(np.single) and FP16(np.half) om model
        self.ndtype = model_ndtype
        self.mode = mode
        self.postprocess_type = postprocess_type
        self.aipp = aipp  
       
        self.model_height, self.model_width = imgsz[0], imgsz[1]  # 图像resize大小
     

    def __call__(self, im0, conf_threshold=0.4, iou_threshold=0.45):
        """
        The whole pipeline: pre-process -> inference -> post-process.

        Args:
            im0 (Numpy.ndarray): original input image.
            conf_threshold (float): confidence threshold for filtering predictions.
            iou_threshold (float): iou threshold for NMS.

        Returns:
            boxes (List): list of bounding boxes.
        """
        # 前处理Pre-process
        t1 = time.time()
        im, ratio, (pad_w, pad_h) = self.preprocess(im0)
        pre_time = round(time.time() - t1, 3)
        
        # 推理 inference
        t2 = time.time()
        preds = self.session.infer([im], mode=self.mode)[0]  # mode有动态"dymshape"和静态"static"等
        det_time = round(time.time() - t2, 3)
        
        # 后处理Post-process
        t3 = time.time()
        if self.postprocess_type == "v5":
            boxes = self.postprocess_v5(preds,
                                    im0=im0,
                                    ratio=ratio,
                                    pad_w=pad_w,
                                    pad_h=pad_h,
                                    conf_threshold=conf_threshold,
                                    iou_threshold=iou_threshold,
                                    )
            
        elif self.postprocess_type == "v8":
            boxes = self.postprocess_v8(preds,
                                    im0=im0,
                                    ratio=ratio,
                                    pad_w=pad_w,
                                    pad_h=pad_h,
                                    conf_threshold=conf_threshold,
                                    iou_threshold=iou_threshold,
                                    )
            
        elif self.postprocess_type == "v10":
            boxes = self.postprocess_v10(preds,
                                    im0=im0,
                                    ratio=ratio,
                                    pad_w=pad_w,
                                    pad_h=pad_h,
                                    conf_threshold=conf_threshold
                                    )
        
        else:
            boxes = []

        post_time = round(time.time() - t3, 3)

        return boxes, (pre_time, det_time, post_time)
        
    # 前处理，包括：resize, pad, 其中HWC to CHW，BGR to RGB，归一化，增加维度CHW -> BCHW可选择是否开启AIPP加速处理
    def preprocess(self, img):
        """
        Pre-processes the input image.

        Args:
            img (Numpy.ndarray): image about to be processed.

        Returns:
            img_process (Numpy.ndarray): image preprocessed for inference.
            ratio (tuple): width, height ratios in letterbox.
            pad_w (float): width padding in letterbox.
            pad_h (float): height padding in letterbox.
        """
        # Resize and pad input image using letterbox() (Borrowed from Ultralytics)
        shape = img.shape[:2]  # original image shape
        new_shape = (self.model_height, self.model_width)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        ratio = r, r
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        pad_w, pad_h = (new_shape[1] - new_unpad[0]) / 2, (new_shape[0] - new_unpad[1]) / 2  # wh padding
        if shape[::-1] != new_unpad:  # resize
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
            
        top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
        left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))  # 填充

        # 是否开启aipp加速预处理，需atc中完成
        if self.aipp:
            return img, ratio, (pad_w, pad_h)
        
        # Transforms: HWC to CHW -> BGR to RGB -> div(255) -> contiguous -> add axis(optional)
        img = np.ascontiguousarray(np.einsum('HWC->CHW', img)[::-1], dtype=self.ndtype) / 255.0
        img_process = img[None] if len(img.shape) == 3 else img
        return img_process, ratio, (pad_w, pad_h)
    
        
    
    # YOLOv5/6/7通用后处理，包括：阈值过滤与NMS
    def postprocess_v5(self, preds, im0, ratio, pad_w, pad_h, conf_threshold, iou_threshold):
        """
        Post-process the prediction.

        Args:
            preds (Numpy.ndarray): predictions come from ort.session.run().
            im0 (Numpy.ndarray): [h, w, c] original input image.
            ratio (tuple): width, height ratios in letterbox.
            pad_w (float): width padding in letterbox.
            pad_h (float): height padding in letterbox.
            conf_threshold (float): conf threshold.
            iou_threshold (float): iou threshold.

        Returns:
            boxes (List): list of bounding boxes.
        """
        # (Batch_size, Num_anchors, xywh_score_conf_cls), v5和v6的[..., 4]是置信度分数，v8v9采用类别里面最大的概率作为置信度score
        x = preds  # outputs: predictions (1, 8400*3, 85)
    
        # Predictions filtering by conf-threshold
        x = x[x[..., 4] > conf_threshold]
       
        # Create a new matrix which merge these(box, score, cls) into one
        # For more details about `numpy.c_()`: https://numpy.org/doc/1.26/reference/generated/numpy.c_.html
        x = np.c_[x[..., :4], x[..., 4], np.argmax(x[..., 5:], axis=-1)]

        # NMS filtering
        # 经过NMS后的值, np.array([[x, y, w, h, conf, cls], ...]), shape=(-1, 4 + 1 + 1)
        x = x[cv2.dnn.NMSBoxes(x[:, :4], x[:, 4], conf_threshold, iou_threshold)]
    
        # 重新缩放边界框，为画图做准备
        if len(x) > 0:
            # Bounding boxes format change: cxcywh -> xyxy
            x[..., [0, 1]] -= x[..., [2, 3]] / 2
            x[..., [2, 3]] += x[..., [0, 1]]

            # Rescales bounding boxes from model shape(model_height, model_width) to the shape of original image
            x[..., :4] -= [pad_w, pad_h, pad_w, pad_h]
            x[..., :4] /= min(ratio)

            # Bounding boxes boundary clamp
            x[..., [0, 2]] = x[:, [0, 2]].clip(0, im0.shape[1])
            x[..., [1, 3]] = x[:, [1, 3]].clip(0, im0.shape[0])

            return x[..., :6]  # boxes
        else:
            return []

    # YOLOv8/9/11通用后处理，包括：阈值过滤与NMS
    def postprocess_v8(self, preds, im0, ratio, pad_w, pad_h, conf_threshold, iou_threshold):
        """
        Post-process the prediction.

        Args:
            preds (Numpy.ndarray): predictions come from ort.session.run().
            im0 (Numpy.ndarray): [h, w, c] original input image.
            ratio (tuple): width, height ratios in letterbox.
            pad_w (float): width padding in letterbox.
            pad_h (float): height padding in letterbox.
            conf_threshold (float): conf threshold.
            iou_threshold (float): iou threshold.

        Returns:
            boxes (List): list of bounding boxes.
        """
        x = preds  # outputs: predictions (1, 84, 8400)
        # Transpose the first output: (Batch_size, xywh_conf_cls, Num_anchors) -> (Batch_size, Num_anchors, xywh_conf_cls)
        x = np.einsum('bcn->bnc', x)  # (1, 8400, 84)
   
        # Predictions filtering by conf-threshold
        x = x[np.amax(x[..., 4:], axis=-1) > conf_threshold]

        # Create a new matrix which merge these(box, score, cls) into one
        # For more details about `numpy.c_()`: https://numpy.org/doc/1.26/reference/generated/numpy.c_.html
        x = np.c_[x[..., :4], np.amax(x[..., 4:], axis=-1), np.argmax(x[..., 4:], axis=-1)]

        # NMS filtering
        # 经过NMS后的值, np.array([[x, y, w, h, conf, cls], ...]), shape=(-1, 4 + 1 + 1)
        x = x[cv2.dnn.NMSBoxes(x[:, :4], x[:, 4], conf_threshold, iou_threshold)]
       
        # 重新缩放边界框，为画图做准备
        if len(x) > 0:
            # Bounding boxes format change: cxcywh -> xyxy
            x[..., [0, 1]] -= x[..., [2, 3]] / 2
            x[..., [2, 3]] += x[..., [0, 1]]

            # Rescales bounding boxes from model shape(model_height, model_width) to the shape of original image
            x[..., :4] -= [pad_w, pad_h, pad_w, pad_h]
            x[..., :4] /= min(ratio)

            # Bounding boxes boundary clamp
            x[..., [0, 2]] = x[:, [0, 2]].clip(0, im0.shape[1])
            x[..., [1, 3]] = x[:, [1, 3]].clip(0, im0.shape[0])

            return x[..., :6]  # boxes
        else:
            return []
    
    # YOLOv10后处理，包括：阈值过滤-无NMS
    def postprocess_v10(self, preds, im0, ratio, pad_w, pad_h, conf_threshold):
        
        x = preds  # outputs: predictions (1, 300, 6) -> (xyxy_conf_cls)
        
        # Predictions filtering by conf-threshold
        x = x[x[..., 4] > conf_threshold]

        # 重新缩放边界框，为画图做准备
        if len(x) > 0:

            # Rescales bounding boxes from model shape(model_height, model_width) to the shape of original image
            x[..., :4] -= [pad_w, pad_h, pad_w, pad_h]
            x[..., :4] /= min(ratio)

            # Bounding boxes boundary clamp
            x[..., [0, 2]] = x[:, [0, 2]].clip(0, im0.shape[1])
            x[..., [1, 3]] = x[:, [1, 3]].clip(0, im0.shape[0])

            return x  # boxes
        else:
            return []


if __name__ == '__main__':
    # Create an argument parser to handle command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--det_model', type=str, default=r"yolov8s.om", help='Path to OM model')
    parser.add_argument('--source', type=str, default=r'images', help='Path to input image')
    parser.add_argument('--out_path', type=str, default=r'results', help='结果保存文件夹')
    parser.add_argument('--imgsz_det', type=tuple, default=(640, 640), help='Image input size')
    parser.add_argument('--classes', type=list, default=['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
            'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
              'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
                  'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
                    'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                      'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                        'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'], help='类别')

    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    parser.add_argument('--iou', type=float, default=0.6, help='NMS IoU threshold')
    parser.add_argument('--device_id', type=int, default=0, help='device id')
    parser.add_argument('--mode', default='static', help='om是动态dymshape或静态static')
    parser.add_argument('--model_ndtype', default=np.single, help='om是fp32或fp16')
    parser.add_argument('--postprocess_type', type=str, default='v8', help='后处理方式, 对应v5/v8/v10三种后处理')
    parser.add_argument('--aipp', default=False, action='store_true', help='是否开启aipp加速YOLO预处理, 需atc中完成om集成')
    args = parser.parse_args()

    # 创建结果保存文件夹
    if not os.path.exists(args.out_path):
        os.mkdir(args.out_path)
    
    print('开始运行：')
    # Build model
    det_model = YOLO(args.det_model, args.imgsz_det, args.device_id, args.model_ndtype, args.mode, args.postprocess_type, args.aipp)
    color_palette = np.random.uniform(0, 255, size=(len(args.classes), 3))  # 为每个类别生成调色板
    
    for i, img_name in enumerate(os.listdir(args.source)):
        try:
            t1 = time.time()
            # Read image by OpenCV
            img = cv2.imread(os.path.join(args.source, img_name))

            # 检测Inference
            boxes, (pre_time, det_time, post_time) = det_model(img, conf_threshold=args.conf, iou_threshold=args.iou)
            print('{}/{} ==>总耗时间: {:.3f}s, 其中, 预处理: {:.3f}s, 推理: {:.3f}s, 后处理: {:.3f}s, 识别{}个目标'.format(i+1, len(os.listdir(args.source)), time.time() - t1, pre_time, det_time, post_time, len(boxes)))

            # Draw rectangles
            for (*box, conf, cls_) in boxes:
                cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                                color_palette[int(cls_)], 2, cv2.LINE_AA)
                cv2.putText(img, f'{args.classes[int(cls_)]}: {conf:.3f}', (int(box[0]), int(box[1] - 9)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                
            cv2.imwrite(os.path.join(args.out_path, img_name), img)
            
        except Exception as e:
            print(e)
    
        