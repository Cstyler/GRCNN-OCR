import os
import cv2

root_dir = '/home/oleynik/project/price_detector/various_pricetags'
save_ext = '.png'

for dirpath, dirnames, filenames in os.walk(root_dir):
    for f in filenames:
        name, ext = os.path.splitext(f)
        if ext == save_ext:
            continue

        img_path = os.path.join(dirpath, f)
        img = cv2.imread(img_path)
        save_path = os.path.join(dirpath, f'{name}{save_ext}')
        cv2.imwrite(save_path, img)

        os.remove(img_path)
        print(img_path, save_path)
