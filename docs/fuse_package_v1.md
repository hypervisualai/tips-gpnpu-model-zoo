# fuse_package_v1 — 모델 패키지 형식

이 저장소의 모델 폴더 하나가 `fuse_package_v1` 패키지 하나다. 이 문서는 이 저장소에 있는 파일(`template.yml`·`variant.yml`·`scale.json`)과 가중치 파일 대조 규칙을 설명한다.

문장 끝의 **"해야 한다"·"안 된다"는 지켜야 하는 규칙**이고, "권장"은 지키지 않아도 틀린 패키지는 아니다.

## 1. 무엇을 담는가

패키지 = 모델을 GPNPU 에서 INT8 로 실행하는 데 필요한 정보. 두 단으로 나뉜다.

| 단 | 파일 | 담는 것 |
|---|---|---|
| 구조 템플릿 | `template.yml` | 같은 구조의 모든 변형이 함께 쓰는 것: 모델을 만드는 법, 필요한 패키지 버전 |
| 변형 | `variant.yml` · `scale.json` · 가중치 파일 | 가중치에 따라 달라지는 것: 클래스 이름, 층별 양자화 범위, 가중치 |

- 변형은 템플릿을 **경로로 가리킨다**(`variant.yml` 의 `template`). 템플릿과 변형은 같은 폴더여도 되고(`template: .`), 다른 폴더여도 된다. 이 저장소는 모델마다 한 폴더에 둘 다 둔다
- 패키지에는 코드를 담지 않는다. 데이터 파일(YAML·JSON·가중치)만 담는다
- 가중치 파일은 패키지 폴더 안에 있어야 하지만 git 에는 넣지 않는다. 받는 곳은 `variant.yml` 의 `weights_url` 에 적는다
- 어느 층을 NPU 로 보낼지는 실행 환경이 정하며 이 저장소에 담지 않는다

## 2. 용어

| 용어 | 뜻 |
|---|---|
| 패키지 | 구조 템플릿 하나 + 그것을 가리키는 변형 하나 |
| 구조 템플릿 (템플릿) | `template.yml` 이 있는 폴더 |
| 변형 | `variant.yml` 이 있는 폴더. 구조는 같고 가중치가 다른 모델 하나(예: yolo26n 구조의 coco·drone) |
| 층 | 모델 안의 `nn.Conv2d` 또는 `nn.Linear` 모듈 하나. 이름은 루트 모듈의 `named_modules()` 이름(BatchNorm fuse 뒤) — [`fuse_scale_v1.md`](fuse_scale_v1.md) §4.4 |
| Cout | 층의 출력 채널 수. `nn.Conv2d` 는 `out_channels`, `nn.Linear` 는 `out_features` |
| imgsz | 정사각 입력의 한 변. 변형의 `scale.json` `imgsz` 가 기준이다 |

## 3. 폴더와 파일 규칙

```
<템플릿>/
  template.yml        필수  §4.1

<변형>/                (템플릿과 같은 폴더여도 된다)
  variant.yml         필수  §4.2
  scale.json          필수  §4.3
  <가중치 파일>         필수  §4.4 (이름 = scale.json weights.file)
```

1. 파일 이름은 위와 같아야 한다. 대소문자를 구별한다
2. YAML 파일은 UTF-8 이다. 값의 해석은 PyYAML `yaml.safe_load` 기준(YAML 1.1)이다
   - 쓰는 쪽: 문자열·정수·목록·맵만 쓴다(`build.args` 의 값만 예외로 실수·bool 도 된다 — §4.1.1). 태그(`!!python/…` 등)·앵커(`&`)·별칭(`*`)·같은 맵 안의 중복 키·`null` 은 안 된다. 선택 칸을 쓰지 않을 때는 칸을 빼고 `null` 을 적지 않는다
   - `on`·`off`·`yes`·`no`·`true`·`false` 처럼 bool 로 읽히는 낱말을 문자열(예: 클래스 이름)로 쓰려면 따옴표로 감싸야 한다
   - 숫자로만 된 문자열은 따옴표로 감싸야 한다. 따옴표가 없으면 정수로 읽힌다
   - 읽는 쪽: 안전 로더로 읽고, 중복 키·앵커·별칭은 오류로 거부해야 한다
3. 각 YAML 파일에는 **이 문서에 정한 칸만** 쓸 수 있다. 정하지 않은 칸이 있으면 오류다. 주석(`#`)은 자유다

## 4. 칸 정의

### 4.1 `template.yml`

| 칸 | 타입 | 필수 | 뜻 |
|---|---|---|---|
| `format` | 문자열 | ● | 반드시 `"fuse_package_v1"` |
| `name` | 문자열 | ● | 구조 이름(예 `"yolo26n"`). 변형 `scale.json` 의 `model` 과 같아야 한다 |
| `build` | 맵 | | 모델을 만드는 법(§4.1.1) |

#### 4.1.1 `build` — 모델을 만드는 법

모델의 구조 코드가 어디에 있느냐에 따라 `kind` 를 고른다.

| `kind` | 구조 코드가 있는 곳 | 가중치 파일에 든 것 | 예 |
|---|---|---|---|
| `weights_file` | 프레임워크. 어느 구조인지는 가중치 파일 안에 있다 | 모델 객체 통째(구조 설정·클래스 수·값) | ultralytics `.pt` |
| `library` | 라이브러리. 패키지의 모델 이름으로 고른다 | 값(state_dict)만 | timm·torchvision |
| `python` | 유저 코드. 패키지의 클래스 위치로 고른다 | 값(state_dict)만 | 유저 패키지의 클래스 |

| 칸 | 타입 | 필수 | 쓰는 `kind` | 뜻 |
|---|---|---|---|---|
| `kind` | `"weights_file"` \| `"library"` \| `"python"` | ● | 모두 | 위 표 |
| `requires` | 문자열 목록 | ● | 모두 | 모델을 만들고 실행하는 데 필요한 패키지. 비어 있으면 안 된다. 항목은 PEP 508 표기(예 `"ultralytics==8.4.35"`, `"timm>=1.0"`). **실행 버전**이다 — 체크포인트를 학습한 버전이 아니다. 패키지를 읽는 쪽은 설치된 버전이 이 조건을 만족하지 않으면 오류를 내야 한다. URL 꼴 항목(`"이름 @ git+https://…"`)은 버전 조건이 없어 설치 여부만 본다 |
| `framework` | 문자열 | ● (`weights_file`·`library`) | `weights_file`·`library` | `weights_file`: `"ultralytics"`. `library`: `"timm"` \| `"torchvision"` |
| `model` | 문자열 | ● (`library`) | `library` | 라이브러리의 모델 이름 |
| `entry` | 문자열 | ● (`python`) | `python` | `"패키지.모듈:클래스"`. import 할 수 있어야 한다 |
| `args` | 맵 | | `library`·`python` | 모델을 만들 때 넘길 고정 인자. 만드는 규칙이 이미 넘기는 이름(`model`·`pretrained`·`weights`)과 `nc_arg` 의 값은 키로 쓸 수 없다 |
| `nc_arg` | 문자열 | ● (`library`·`python`) | `library`·`python` | 클래스 수를 받는 인자 이름. 값은 변형 `classes` 개수로 채운다 |
| `state_dict_key` | 문자열 | | `library`·`python` | 가중치 파일이 `{키: state_dict}` 처럼 싸여 있을 때 그 키 |

`kind` 에 쓰지 않는 칸이 있으면 오류다.

만드는 규칙

| `kind` | 규칙 |
|---|---|
| `weights_file` (`ultralytics`) | `YOLO(<가중치>).model`. 그다음 `len(classes) == model.nc` 를 확인하고 다르면 오류 |
| `library` (`timm`) | `timm.create_model(model, pretrained=False, **args, <nc_arg>=len(classes))` |
| `library` (`torchvision`) | `torchvision.models.get_model(model, weights=None, **args, <nc_arg>=len(classes))` |
| `python` | `<entry 의 클래스>(**args, <nc_arg>=len(classes))` |
| `library`·`python` 공통 | 그다음 `load_state_dict(<state_dict>, strict=True)`. state_dict = `torch.load(<가중치>, map_location="cpu", weights_only=True)`, `state_dict_key` 가 있으면 그 키의 값. 어긋나는 키·형상이 있으면 오류 |

만든 결과가 층 이름의 기준이 되는 **루트 모듈**이다.

열린 어휘 모델(실행 중에 텍스트로 클래스를 정하는 모델)은 v1 패키지로 만들지 않는다 — `classes` 에 무엇을 적을지 정해지지 않았다.

### 4.2 `variant.yml`

| 칸 | 타입 | 필수 | 뜻 |
|---|---|---|---|
| `format` | 문자열 | ● | 반드시 `"fuse_package_v1"` |
| `template` | 문자열 | ● | 템플릿 폴더 경로. 상대 경로면 **이 변형 폴더 기준**이다. 같은 폴더면 `.` 이다. 그 폴더에 `template.yml` 이 있어야 한다 |
| `classes` | 문자열 목록 | ● | 클래스 이름. 순서 = 모델 출력의 클래스 번호. 비어 있으면 안 된다. 개수 = 클래스 수 |
| `weights_url` | 문자열 | | 가중치 파일을 받는 곳. 읽는 쪽은 받아 오지 않는다 — 가중치 파일이 없을 때 알려 주는 데 쓴다 |

- 변형 이름은 `scale.json` 의 `variant` 다. `variant.yml` 에 따로 적지 않는다
- imgsz·가중치 파일 이름·sha256 은 `scale.json` 에 있다. `variant.yml` 에 다시 적지 않는다

### 4.3 `scale.json`

[`fuse_scale_v1.md`](fuse_scale_v1.md) 를 그대로 따른다. 패키지 안에서는 다음을 더 지켜야 한다.

1. 파일 이름은 `scale.json` 이다
2. `model` = 템플릿 `name`
3. NPU 로 보내는 층은 모두 `layers` 에 있어야 한다

### 4.4 가중치 파일

- 변형 폴더 안에 `scale.json` `weights.file` 이름으로 있어야 한다
- 내용이 템플릿 `build` 의 `kind` 에 맞아야 한다(§4.1.1)
- sha256 = `scale.json` `weights.sha256`
- 권장: git 에 넣지 않는다. 받는 곳은 `weights_url` 에 적는다

## 5. 검사 규칙

어기면 모두 **오류**다(읽는 쪽은 거부하고 멈춘다).

### 5.1 파일 검사 (패키지 파일만으로 한다)

- §3 파일 이름·YAML 규칙, §4 칸 이름·타입·필수 여부, 정하지 않은 칸이 없는지 — `template.yml`(`build` 포함)·`variant.yml`
- `template.yml`·`variant.yml` 의 `format` = `"fuse_package_v1"`
- `build.requires` 의 각 항목이 PEP 508 요구 사항으로 읽히고, 설치된 패키지가 모두 만족한다(버전 비교는 PEP 440)
- `variant.yml` `template` 경로에 `template.yml` 이 있다
- `scale.json` 이 fuse_scale_v1 검사(그 문서 §6.1·§6.2)를 통과한다
- `scale.json` `model` = 템플릿 `name`
- 가중치 파일이 있고 sha256 = `scale.json` `weights.sha256` (없으면 `weights_url` 을 알려 준다)

### 5.2 모델 검사 (만들고 BatchNorm 을 fuse 한 모델로 한다)

- NPU 로 보내는 층이 모두 모델에 있다
- 그 층의 `scale.json` `op` 와 모듈 종류가 맞다: `conv2d` ↔ `nn.Conv2d`, `linear` ↔ `nn.Linear`
- 그 층의 `out.max` 가 배열이면 길이 = 그 모듈의 Cout. 값 하나면 Cout 만큼 같은 값으로 펼친다

## 6. 버전 규칙

- v1 에는 칸·파일을 더할 수 없다. 더하거나 뜻을 바꾸면 새 형식 이름(`fuse_package_v2` …)을 쓴다
- 읽는 쪽은 아는 형식 이름만 받고 나머지는 거부해야 한다
- fuse_package_v1 은 `scale.json` 으로 fuse_scale_v1 만 받는다

## 7. 예시 — `yolo26n_drone`

`template.yml`

```yaml
format: fuse_package_v1
name: yolo26n
build:
  kind: weights_file
  framework: ultralytics
  requires: ["ultralytics==8.4.35"]
```

`variant.yml`

```yaml
format: fuse_package_v1
template: .
weights_url: https://github.com/hypervisualai/tips-gpnpu-model-zoo/releases/download/weights-v1/yolo26_drone.pt
classes: [pedestrian, people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor]
```

`scale.json` — fuse_scale_v1 (`model: "yolo26n"`, `variant: "drone"`, `imgsz: 320`, `weights.file: "yolo26_drone.pt"`)

`yolo26_drone.pt` — ultralytics 가중치(클래스 10개). git 에는 없고 `weights_url` 에서 받는다

`library` 의 예 — `swin_t_imagenet/template.yml`

```yaml
format: fuse_package_v1
name: swin_t
build:
  kind: library
  framework: timm
  model: swin_tiny_patch4_window7_224
  nc_arg: num_classes
  requires: ["timm>=1.0"]
```
