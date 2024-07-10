import argparse
import time
from pathlib import Path

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel

import os
import mariadb
# import MySQLdb
# import mysql.connector

import pyodbc
# import textdistance
import difflib

# Untuk Super Resolution
from cv2 import dnn_superres
sr = dnn_superres.DnnSuperResImpl_create()
path = "./FSRCNN_x4.pb"
sr.readModel(path)
sr.setModel("fsrcnn", 4)


#> Untuk paddle ORC
from paddleocr import PaddleOCR,draw_ocr
ocr = PaddleOCR(use_angle_cls=True, lang='en')
import re


def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    # save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    save_dir = Path('./runs/detect')
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    # Number plate object
                    x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    box = im0[int(y1)-15:int(y2)+15, int(x1)-15:int(x2)+15]

                    # Super Resolution
                    try:
                        box = sr.upsample(box)
                    except:
                        pass
                    # cv2.imshow("roi", box)
                    # cv2.waitKey(0)
                    # cv2.imwrite(save_path, box)
                    
                    # OCR
                    plate_num = ""
                    try:
                        result = ocr.ocr(box, cls=True)
                        for i in result:
                            platenum =[line[1][0] for line in i]
                            for text in platenum:
                                clean_text = re.sub('[\W_]+', '', text)
                                plate_num += str(clean_text)
                    except:
                        text = None
                    
                    # treatment for indonesia plate number format
                    if len(plate_num) >= 9:
                        try:
                            int(plate_num[8])
                            plate_num = plate_num[0:8]
                        except:
                            plate_num = plate_num[0:9]
                    else:
                        plate_num=plate_num

                    # cek similarity plat no di database (if > 90% ambil yang di database, if < 90% ambil yang ocr)
                    driver_db = "ODBC Driver 17 for SQL Server"
                    server_db = "xxx.xxx.xxx.xxx"
                    dbname = "xxxx"
                    username = "xx"
                    password = "xxxxx"
                    timeout = 1

                    plate_num_predict = plate_num
                    try:
                        connection_string = f'DRIVER={driver_db};SERVER={server_db};DATABASE={dbname};UID={username};PWD={password};Connect Timeout={timeout}'
                        conn = pyodbc.connect(connection_string)
                        cursor = conn.cursor()
                        
                        # Check if plate_num_predict exists in the database
                        query_check = "SELECT COUNT(*) FROM [xxxxx].[dbo].[VEHICLEMASTER] WHERE [number_plat] = ?"
                        cursor.execute(query_check, (plate_num_predict,))
                        if cursor.fetchone()[0] > 0:
                            conn.close()  # Close connection if already exists
                        else:
                            query = "SELECT [number_plat] FROM [xxxxx].[dbo].[VEHICLEMASTER]"
                            cursor.execute(query)

                            similarity_threshold = 0.9
                            best_similarity = 0

                            # Compare OCR result with each [number_plat] and find the most similar one
                            for row in cursor.fetchall():
                                db_plat = row[0]
                                # similarity = textdistance.jaro_winkler(plate_num, db_plat)
                                similarity = difflib.SequenceMatcher(None, plate_num, db_plat).ratio()
                                
                                if similarity > similarity_threshold and similarity > best_similarity:
                                    plate_num_predict = db_plat
                                    best_similarity = similarity

                            conn.close()
                    except:
                        print("gagal mengecek data di database quota central")

                    # put bbox and plat on image
                    if save_img or view_img:  # Add bbox to image
                        label = f'{names[int(cls)]} {conf:.2f}'
                        plot_one_box(xyxy, im0, label=plate_num_predict, color=[0, 0, 255], line_thickness=2) # just plat and bbox
                        # plot_one_box(xyxy, im0, label=plate_num+" "+f'({time.time() - t0:.3f}s)', color=[0, 0, 255], line_thickness=2)  # with computation duration

                    # Save to database
                    # Split file name to get site_id, disp_no, nozzel_no
                    file_name_parts = os.path.splitext(os.path.basename(path))[0].split('_')
                    site_id = file_name_parts[0]
                    disp_no = int(file_name_parts[1])
                    nozzel_no = int(file_name_parts[2])
                    save_to_database(site_id, disp_no, nozzel_no, plate_num_predict)

                    # Delete the original image
                    os.remove(path)
            else:    
                # Delete the original image
                os.remove(path)
                print("No plat tidak terdeteksi")

            # Print time (inference + NMS)
            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')

            # Stream results
            if view_img:
                cv2.imshow(str(p), im0)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, im0.shape[1], im0.shape[0]
                            save_path += '.mp4'
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(im0)

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')

def save_to_database(site_id, disp_no, nozzel_no, plate_num):
    try:
        # Establish a connection to the database
        conn = mariadb.connect(
            user="xxxx",
            password="xxxx",
            host="xxx.xxx.xxx.xxx",
            port=3306,
            database="xxxx"
        )
        
        # Create a cursor to execute SQL queries
        cursor = conn.cursor()

        # Construct the SQL query
        sql = "INSERT INTO imagetoplatno (siteID, disp_no, nozzel_no, plat_no, status) VALUES (?, ?, ?, ?, ?)"
        values = (site_id, disp_no, nozzel_no, plate_num, 0)

        # Execute the SQL query
        cursor.execute(sql, values)

        # Commit changes and close the connection
        conn.commit()
        conn.close()

    except mariadb.Error as e:
        print(f"Error connecting to MariaDB: {e}")

# def save_to_database(site_id, disp_no, nozzel_no, plate_num):
#     try:
#         # Establish a connection to the database
#         conn = MySQLdb.connect(
#             user="root",
#             passwd="minic123",
#             host="localhost",
#             port=3306,
#             db="anpr"
#         )
        
#         # Create a cursor to execute SQL queries
#         cursor = conn.cursor()

#         # Construct the SQL query
#         sql = "INSERT INTO imagetoplatno (siteID, disp_no, nozzel_no, plat_no, status) VALUES (%s, %s, %s, %s, %s)"
#         values = (site_id, disp_no, nozzel_no, plate_num, 0)

#         # Execute the SQL query
#         cursor.execute(sql, values)

#         # Commit changes and close the connection
#         conn.commit()
#         conn.close()

#     except MySQLdb.Error as e:
#         print("Error:", e)

# def save_to_database(site_id, disp_no, nozzel_no, plate_num):
#     try:
#         # Establish a connection to the database
#         conn = mysql.connector.connect(
#             user="root",
#             password="minic123",
#             host="localhost",
#             port=3306,
#             database="anpr"
#         )
        
#         # Create a cursor to execute SQL queries
#         cursor = conn.cursor()

#         # Construct the SQL query
#         sql = "INSERT INTO imagetoplatno (siteID, disp_no, nozzel_no, plat_no, status) VALUES (%s, %s, %s, %s, %s)"
#         values = (site_id, disp_no, nozzel_no, plate_num, 0)

#         # Execute the SQL query
#         cursor.execute(sql, values)

#         # Commit changes and close the connection
#         conn.commit()
#         conn.close()

#     except mysql.connector.Error as e:
#         print("Error:", e)

def detect_folder_images(folder_path, opt):
    while True:
        # List all files in the folder
        image_files = os.listdir(folder_path)
        
        if not image_files:
            # No images to process, sleep for a while before checking again
            time.sleep(5)
            continue

        for image_file in image_files:
            image_path = os.path.join(folder_path, image_file)
            opt.source = image_path  # Set the image path as the source

            # Extract the side (L or R) from the image file name
            side = image_file.split('_')[-2].split('.')[0].upper()

            # Load the image
            image = cv2.imread(image_path)

            # Get the width and height of the image
            height, width, _ = image.shape

            # Calculate the center of the image
            center = width // 2

            # Crop the image based on the side
            if side == 'L':
                cropped_image = image[:, :center]
            elif side == 'R':
                cropped_image = image[:, center:]
            elif side == 'N':
                cropped_image = image
            else:
                # If the side is not 'L' or 'R', skip this image and continue to the next one
                print(f"Skipping {image_path}. Invalid side: {side}")
                continue

            # Save the cropped image, overwriting the original image file
            cv2.imwrite(image_path, cropped_image)

            # Now you can perform any additional task you want with the cropped image
            # For example, call the detect function with the cropped image
            detect(opt)  # Assuming detect function takes opt and processed the cropped image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='yolov7.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect(opt)
                strip_optimizer(opt.weights)
        else:
            # Set the appropriate path to the image_try folder
            folder_path = './cctvCapture'
            detect_folder_images(folder_path, opt)

