# TIPS GPNPU Model Zoo

HyperVisual AI 의 GPNPU(GPGPU + NPU 하이브리드 프로세서)에서 모델을 INT8 로 실행하는 데 쓰는 **모델 패키지** 모음이다.
패키지 하나는 모델 하나에 대해 "모델을 어떻게 만드는가 · 클래스 이름 · 층별 INT8 양자화 범위 · 어느 가중치 파일에 맞는가"를 담는다.

- 패키지는 데이터 파일(YAML·JSON)만 담는다. 코드와 가중치 파일은 담지 않는다
- 가중치는 각 패키지의 `variant.yml` `weights_url` 에서 받는다
- 패키지를 읽어 실행하는 SDK(런타임·드라이버 포함)는 이 저장소에서 공개하지 않는다

## 모델

| 폴더 | 모델 | 분야 | 클래스 수 | 입력 크기 | 가중치 받는 곳 | 가중치 sha256 |
|---|---|---|---|---|---|---|
| [`yolo26n_coco`](yolo26n_coco/) | YOLO26n | 객체 검출 | 80 (COCO) | 320 | [ultralytics/assets v8.4.0](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt) | `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef` |
| [`yolo26n_drone`](yolo26n_drone/) | YOLO26n (드론 영상) | 객체 검출 | 10 (VisDrone) | 320 | [이 저장소 Release `weights-v1`](https://github.com/hypervisualai/tips-gpnpu-model-zoo/releases/download/weights-v1/yolo26_drone.pt) | `f5f27b81d36b46c64df1605f5eb888bb61e84b148e0be2aae7ebd005f9c0d0b5` |
| [`swin_t_imagenet`](swin_t_imagenet/) | Swin-T | 분류 | 1000 (ImageNet-1k) | 224 | [이 저장소 Release `weights-v1`](https://github.com/hypervisualai/tips-gpnpu-model-zoo/releases/download/weights-v1/swin_tiny_patch4_window7_224.pth) | `8d4fbe034d7934760804e9f3d8bde609c7ffff44322036f4c827e165b924a678` |
| [`yolo11s_coco`](yolo11s_coco/) | YOLO11s | 객체 검출 | 80 (COCO) | 320 | [ultralytics/assets v8.4.0](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo11s.pt) | `85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5` |
| [`yolo26s_coco`](yolo26s_coco/) | YOLO26s | 객체 검출 | 80 (COCO) | 320 | [ultralytics/assets v8.4.0](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26s.pt) | `646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b` |
| [`yolov8s_coco`](yolov8s_coco/) | YOLOv8s | 객체 검출 | 80 (COCO) | 320 | [ultralytics/assets v8.4.0](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8s.pt) | `1f47a78bf100391c2a140b7ac73a1caae18c32779be7d310658112f7ac9aa78a` |
| [`yolov13n_coco`](yolov13n_coco/) | YOLOv13-N | 객체 검출 | 80 (COCO) | 320 | [iMoonLab/yolov13](https://github.com/iMoonLab/yolov13/releases/download/yolov13/yolov13n.pt) | `6653035017b0f111f80ec11ed914874ea85699b104aeac1e46e517d16889d6b7` |
| [`yolo26n_seg_coco`](yolo26n_seg_coco/) | YOLO26n-seg | 인스턴스 분할 | 80 (COCO) | 320 | [ultralytics/assets v8.4.0](https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n-seg.pt) | `361fbfabab285c3237700b6bb91d7ecfa602cd945fffda8dbe1242829b71e73f` |

- 입력 크기 = 정사각 입력의 한 변(픽셀). 양자화 범위는 이 크기에서 쟀으므로 이 크기로 실행할 때만 맞는다
- 공식 배포처의 가중치는 배포 파일 그대로이고, 위 sha256 은 그 파일의 값이다
- 드론 YOLO26n 은 VisDrone2019-DET 로 자체 학습한 가중치다. Release 파일은 학습 기록 칸(학습 설정·지표·옵티마이저 등)을 지운 것이고 가중치 값은 그대로다. Swin-T 는 timm `swin_tiny_patch4_window7_224`(ImageNet-1k) 공개 가중치를 timm state_dict 로 저장한 파일이다

## 폴더

```
<모델 폴더>/
  template.yml   모델 구조 — 모델을 만드는 법과 실행에 필요한 패키지 버전
  variant.yml    클래스 이름, 가중치 받는 곳(weights_url)
  scale.json     층별 INT8 양자화 범위, 맞는 가중치 파일의 이름·sha256
docs/
  fuse_package_v1.md            패키지 형식
  fuse_scale_v1.md              scale.json 형식
  fuse_scale_v1.schema.json     scale.json 의 JSON Schema
```

- 모델 폴더 하나가 패키지 하나다. `variant.yml` 의 `template: .` 은 같은 폴더의 `template.yml` 을 쓴다는 뜻이다
- `yolo26n_coco` 와 `yolo26n_drone` 은 구조가 같아 `template.yml` 내용이 같다

## 가중치 받기와 확인

가중치를 `weights_url` 에서 받아 패키지 폴더 안에 `scale.json` 의 `weights.file` 이름으로 둔다. 그다음 sha256 이 `scale.json` 의 `weights.sha256` 과 같은지 확인한다.

```bash
sha256sum yolo26n_coco/yolo26n.pt     # = yolo26n_coco/scale.json 의 weights.sha256
```

`scale.json` 의 값은 그 sha256 의 가중치에만 맞는다. 다른 가중치(다시 학습했거나 다른 판)에는 쓸 수 없다.

## 참고

- **Swin-T** — 이 저장소의 `swin_t_imagenet/scale.json` 과 공개 가중치만으로 ImageNet-1k val 50,000장에서 top-1 80.99 %(FP32 대비 −0.39 %p)를 확인했다
- `scale.json` 의 `calib` 칸은 값을 만든 도구·데이터를 적어 둔 기록이다. 실행에는 쓰이지 않는다
- 형식을 직접 검사하려면 `docs/fuse_scale_v1.md` 의 "검사하는 법" — JSON Schema 로 `scale.json` 을 검사할 수 있다
