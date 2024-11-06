# %%

from PIL import Image
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms




def visualize_results(boxed_img, tensor_img):
    plt.figure(figsize=(10,5))
    # PIL 이미지로 변환하여 Jupyter 노트북에서 표시
    display_image = Image.fromarray(boxed_img)
    plt.subplot(1,2,1)
    plt.title('detected_image')
    plt.imshow(display_image)
    plt.axis('off')

    visualized_img = tensor_img.permute(1,2,0).numpy()
    h, w, _ = visualized_img.shape
    plt.subplot(1,2,2)
    plt.imshow(visualized_img)
    plt.title(f'Output Size: {visualized_img.shape[0]}x{visualized_img.shape[1]}')
    plt.axis('off')
    plt.show()


def preprocess_image(img, mtcnn,size=224, visualize=False):
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    # 1. 얼굴 검출
    boxes, _ = mtcnn.detect(pil_img)
    
    # 1-1. 얼굴 검출 결과 표시
    if boxes is not None:
        boxed_img = img.copy()
        h, w, _ = img.shape
        # 이미지의 중심점 계산
        img_center = np.array([w / 2, h / 2])

        # 얼굴 중심점 계산 및 중앙과의 거리 계산
        closest_box = min(boxes, key=lambda box: np.linalg.norm(img_center - np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])))
        x1, y1, x2, y2 = map(int, closest_box)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
        # 얼굴 영역 표시
        cv2.rectangle(boxed_img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        # 얼굴영역 cropping
        roi = img[y1:y2, x1:x2]
        # 2. resizing
        resized_img = cv2.resize(roi, (size,size))
        # 3. Normalization
        normalized_image = resized_img / 255.0
        # 4. Transforms to tensor -> bytebuffer
        tensor_img = transforms.ToTensor()(normalized_image)
        # byte_buffer = tensor_img.numpy().tobytes()
        if visualize:
            visualize_results(boxed_img, tensor_img)
        return tensor_img
    else:
        print("얼굴이 검출되지 않았습니다.")
        return None
# %%
