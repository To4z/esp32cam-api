import cv2
import torch
import re
import function.utils_rotate as utils_rotate
import function.helper as helper

# Hàm chuẩn hóa định dạng biển số
def smart_format_plate(raw):
    if re.match(r'^\d{2}[A-Z]\d{5}$', raw):
        return f"{raw[:3]}-{raw[3:6]}.{raw[6:]}"
    if re.match(r'^\d{2}[A-Z]{1,2}\d{5}$', raw):
        return f"{raw[:2]}-{raw[2:4]} {raw[4:7]}.{raw[7:]}"
    return raw

# Load model YOLO (dùng source từ GitHub)
yolo_LP_detect = torch.hub.load('ultralytics/yolov5', 'custom', path='model/LP_detector.pt', force_reload=False)
yolo_license_plate = torch.hub.load('ultralytics/yolov5', 'custom', path='model/LP_ocr.pt', force_reload=False)
yolo_license_plate.conf = 0.60

# Hàm nhận diện biển số từ ảnh
def detect_plates_from_image(img):
    list_read_plates = set()
    plates = yolo_LP_detect(img, size=640)
    list_plates = plates.pandas().xyxy[0].values.tolist()

    if len(list_plates) == 0:
        lp = helper.read_plate(yolo_license_plate, img)
        if lp != "unknown":
            lp = smart_format_plate(lp)
            list_read_plates.add(lp)
    else:
        for plate in list_plates:
            flag = 0
            x = int(plate[0])
            y = int(plate[1])
            w = int(plate[2] - plate[0])
            h = int(plate[3] - plate[1])
            crop_img = img[y:y+h, x:x+w]

            for cc in range(0, 2):
                for ct in range(0, 2):
                    lp = helper.read_plate(yolo_license_plate, utils_rotate.deskew(crop_img, cc, ct))
                    if lp != "unknown":
                        lp = smart_format_plate(lp)
                        list_read_plates.add(lp)
                        flag = 1
                        break
                if flag == 1:
                    break

    return list(list_read_plates)
