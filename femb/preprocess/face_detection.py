# %%
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import random
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms as transforms


def preprocess_image(img, size=224):
    # MTCNN 모델 생성
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mtcnn = MTCNN(keep_all=True, device=device)
    # 얼굴 임베딩 모델
    model = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    # 이미지 변환
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img)
    # 1. 얼굴 검출
    boxes, _ = mtcnn.detect(pil_img)

    tensor_imgs = []
    byte_buffers = []

    # 1-1. 얼굴 검출 결과 표시
    if boxes is not None:
        boxed_img = img.copy()
        for box in boxes:
            # box는 [x1, y1, x2, y2] 형식의 좌표를 가진다.
            x1, y1, x2, y2 = map(int, box)
            # 음수 좌표가 있을 경우 0으로 변경
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
            # cv2로 사각형 그리기 (BGR 형식으로 빨간색 사용)
            cv2.rectangle(boxed_img, (x1, y1), (x2, y2), (255, 0, 0), 3)
            # 얼굴영역 cropping
            roi = img[y1:y2, x1:x2]
            # 2. resizing
            resized_img = cv2.resize(roi, (size,size))
            # 3. Normalization
            normalized_image = resized_img / 255.0
            # 4. Transforms to tensor -> bytebuffer
            to_tensor = transforms.Compose([
                transforms.ToTensor(),
            ])
            tensor_img = to_tensor(normalized_image)
            byte_buffer = tensor_img.numpy().tobytes()

            tensor_imgs.append(tensor_img)
            byte_buffers.append(byte_buffer)

        for tensor_img in tensor_imgs:
            visualized_img = tensor_img.permute(1,2,0)
            shape_img = visualized_img.shape
            plt.figure()
            plt.imshow(visualized_img)
            plt.title(shape_img)
            plt.show()

        # PIL 이미지로 변환하여 Jupyter 노트북에서 표시
        display_image = Image.fromarray(boxed_img)
        plt.title('detected_image')
        plt.imshow(display_image)
        plt.show()
    else:
        print("얼굴이 검출되지 않았습니다.")
    return tensor_imgs



# # 테스트
# for i in random.sample(range(0,13237),10):
#     img_path = lfw_people[i]
#     tensor_img, _ = preprocess_image(img_path, size=96)

# %%
