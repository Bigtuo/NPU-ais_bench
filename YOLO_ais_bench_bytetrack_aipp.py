import argparse
import time 
import cv2
import numpy as np
import os
import copy

from ais_bench.infer.interface import InferSession
from bytetrack.byte_tracker import BYTETracker


class YOLO:
    """YOLO object detection model class for handling inference and visualization."""

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
    
    # 绘框
    def draw_and_visualize(self, im, bboxes, video_writer, classes, color_palette, vis=False, save=False, is_track=False):
        """
        Draw and visualize results.

        Args:
            im (np.ndarray): original image, shape [h, w, c].
            bboxes (numpy.ndarray): [n, 6], n is number of bboxes.
            vis (bool): imshow using OpenCV.
            save (bool): save image annotated.

        Returns:
            None
        """
        # Draw rectangles 
        if not is_track:
            for (*box, conf, cls_) in bboxes:
                # draw bbox rectangle
                cv2.rectangle(im, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                            color_palette[int(cls_)], 1, cv2.LINE_AA)
                cv2.putText(im, f'{classes[int(cls_)]}: {conf:.3f}', (int(box[0]), int(box[1] - 9)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_palette[int(cls_)], 2, cv2.LINE_AA)
        else:
            for (*box, conf, id_) in bboxes:
                # draw bbox rectangle
                cv2.rectangle(im, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])),
                            (0, 0, 255), 1, cv2.LINE_AA)
                cv2.putText(im, f'{id_}: {conf:.3f}', (int(box[0]), int(box[1] - 9)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    
        # Show image
        if vis:
            cv2.imshow('demo', im)
            cv2.waitKey(1)

        # Save video
        if save:
            video_writer.write(im)


class ByteTrackerONNX(object):
    def __init__(self, args):
        self.args = args
        self.tracker = BYTETracker(args, frame_rate=30)

    def _tracker_update(self, dets, image):
        online_targets = []
        if dets is not None:
            online_targets = self.tracker.update(
                dets[:, :5],
                [image.shape[0], image.shape[1]],
                [image.shape[0], image.shape[1]],
            )

        online_tlwhs = []
        online_ids = []
        online_scores = []
        for online_target in online_targets:
            tlwh = online_target.tlwh
            track_id = online_target.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6
            if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(track_id)
                online_scores.append(online_target.score)

        return online_tlwhs, online_ids, online_scores
    
    
    def inference(self, image, dets):
        """
        Args: dets: 检测结果, [x1, y1, x2, y2, conf, cls]
        Returns: np.array([[x1, y1, x2, y2, conf, ids], ...])
        """
        bboxes, ids, scores = self._tracker_update(dets, image)
        if len(bboxes) == 0:
            return []
        # Bounding boxes format change: tlwh -> xyxy
        bboxes = np.array(bboxes)
        bboxes[..., [2, 3]] += bboxes[..., [0, 1]]
        bboxes = np.c_[bboxes, np.array(scores), np.array(ids)]
        return bboxes
    

if __name__ == '__main__':
    # Create an argument parser to handle command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--det_model', type=str, default=r"yolov8s.om", help='Path to OM model')
    parser.add_argument('--source', type=str, default=r'test.mp4', help='Path to input video')
    parser.add_argument('--out_path', type=str, default=r'res.mp4', help='结果保存文件')
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

    parser.add_argument('--is_track', default=False, action='store_true', help='是否启用跟踪')
    parser.add_argument('--track_thresh', type=float, default=0.5, help='tracking confidence threshold')
    parser.add_argument('--track_buffer', type=int, default=30, help='the frames for keep lost tracks, usually as same with FPS')
    parser.add_argument('--match_thresh', type=float, default=0.8, help='matching threshold for tracking')
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes',)
    parser.add_argument('--mot20', dest='mot20', default=False, action='store_true', help='test mot20.',)
    args = parser.parse_args()
    
    print('开始运行：')
    # Build model
    det_model = YOLO(args.det_model, args.imgsz_det, args.device_id, args.model_ndtype, args.mode, args.postprocess_type, args.aipp)
    bytetrack = ByteTrackerONNX(args)
    color_palette = np.random.uniform(0, 255, size=(len(args.classes), 3))  # 为每个类别生成调色板
    
    # 读取视频,解析帧数宽高,保存视频
    cap = cv2.VideoCapture(args.source)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    video_writer = cv2.VideoWriter(str(args.out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (int(width), int(height)))
    frame_id = 1

    while True:
        start_time = time.time()
        ret, img = cap.read()
        if not ret:
            break

        # Inference
        boxes, (pre_time, det_time, post_time) = det_model(img, conf_threshold=args.conf, iou_threshold=args.iou)
        print('预处理: {:.3f}s, 推理: {:.3f}s, 后处理: {:.3f}s, 识别{}个目标'.format(pre_time, det_time, post_time, len(boxes)))
        
        # track
        if args.is_track:
            track_time = time.time()
            boxes = bytetrack.inference(img, boxes)
            print('跟踪耗时: {:.3f}s'.format(time.time() - track_time))

        # Visualize
        if len(boxes) > 0:
            det_model.draw_and_visualize(copy.deepcopy(img), boxes, video_writer, args.classes, color_palette, vis=False, save=True, is_track=args.is_track)
        
        end_time = time.time() - start_time
        print('frame {}/{} (Total time: {:.2f} ms)'.format(frame_id, int(frame_count), end_time * 1000))
        frame_id += 1
    
        