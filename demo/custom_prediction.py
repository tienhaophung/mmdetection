from argparse import ArgumentParser
import os
from os import path as osp
import json

from mmdet.apis import inference_detector, init_detector, show_result_pyplot

def list_dir(data_dir, allowed_extensions=['.json']):
    """
    List files in directory
    Args:
        data_dir: data directory
        allowed_extensions: File extensions were accepted
    Returns:
        file_paths: list of files
    """
    file_paths = []
    # anot_paths = []
    # List files in data_dir
    for root, dirs, files in os.walk(data_dir):
        # print(root, dirs, files)
        for file in files:
            filename, extension = os.path.splitext(file)
            if extension in allowed_extensions:
                file_paths.append(os.path.join(root, file))
    
    # print(file_paths)
    return file_paths

def makedir(path):
    try:
        os.makedirs(path)
        print("Directory %s created successfully!" %path)
    except OSError:
        print("Directory %s failed to create!" %path)

'''
    [{
        "version": "1.0",
        "image": {
            "folder": "/export/guanghan/Data_2018/posetrack_data/images/val/000342_mpii_test",
            "name": "000000.jpg",
            "id": 0
        },
        "candidates": [
            {
                "det_bbox": [
                    228.1736297607422,
                    123.06813049316406,
                    481.76197814941406,
                    204.4611053466797
                ],
                "det_score": 0.9851313233375549
            },
    },(...)
    ]

'''

def main():
    parser = ArgumentParser()
    parser.add_argument('img', help='Image file')
    parser.add_argument('config', help='Config file')
    parser.add_argument('checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    parser.add_argument(
        '--score-thr', type=float, default=0.3, help='bbox score threshold')
    parser.add_argument("--annot_dir", type=str, default="",
                    help="Annotation directory")
    parser.add_argument("--output_dir", type=str, default="",
                    help="Output directory")
    args = parser.parse_args()

    anot_paths = list_dir(args["annot_dir"])
    anot_paths.sort()
    makedir(args["output_dir"])
    

    num_frames_list = []

    # build the model from a config file and a checkpoint file
    model = init_detector(args.config, args.checkpoint, device=args.device)

    for video_id, path in enumerate(anot_paths):
        # ---------------------- Get video filename and its anotation ----------------------------
        print("Groundtruth anotation file:", path)
        with open(path, 'rt') as json_f:
            video_anot = json.load(json_f)
        
        # Get annotation's filename
        anot_fn = os.path.split(path)[1]
        anot_file = os.path.join(args["output_dir"], anot_fn)
        # print(anot_fn)
        num_frames = len(video_anot["annolist"])
        print("#Frames: %d" %num_frames)
        num_frames_list.append(num_frames)

        # Get video's filename
        video_fn = os.path.split(video_anot["annolist"][0]["image"][0]["name"])[0]
        video_file = args["data_dir"] + '/' + video_fn
        print("Video: %s" %video_fn)
        
        video_list = list_dir(video_fn, ['.jpg', '.jpeg', '.png'])
        video_list.sort()

        output_data = []

        for frame_id, frame_path in enumerate(video_list):
            img_dict = {"version": "1.0",
                        "image": {
                            "folder": video_fn,
                            "name": osp.split(frame_path)[1],
                            "id": frame_id
                        },
                        "candidates": []
            }
            
            candidates = []
            # test a single image
            result = inference_detector(model, args.img) # bboxes: (#classes, #objects, 5), 
            # 5 values: x1, y1, x2, y2, conf

            human_bboxes = result[0] # human catergory: 0 (according coco format)
            
            for box in human_bboxes:
                if box[-1] > args["score-thr"]:
                    x, y = box[0], box[1]
                    w, h = box[2] - box[0], box[3] - box[1]
                    box_dict = {"det_bbox": [x, y, w, h],
                                "det_score": box[-1]}
                    
                    candidates.append(box_dict)
            
            # Store detections
            img_dict["candidates"] = candidates
            output_data.append(img_dict)

        # Store json for video
        with open(anot_file, "wt") as f:
            json.dump(output_data, f, indent=4)
        
        print()

    print("#videos: %d" %len(anot_paths))
    print("Frame range: {} - {}".format(min(num_frames_list), max(num_frames_list)))
    # # show the results
    # show_result_pyplot(model, args.img, result, score_thr=args.score_thr)


if __name__ == '__main__':
    main()
