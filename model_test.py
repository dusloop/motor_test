# integrated_model_test.py
# 웹캠 재질 예측 + 화면/AI 추론 완전 분리 버전
#
# 같은 폴더에 필요한 파일:
#   main_script.py
#   best_epoch012_val0.6905.pt
#
# 실행: python integrated_model_test.py
# 종료: 화면에서 q 키
#
# 핵심 개선:
# 1) 카메라 캡처 스레드
# 2) AI 추론 스레드
# 3) 화면 표시 루프는 추론을 기다리지 않음
# 4) CPU 스레드 수 제한, 추론 간격 조절, 결과 스무딩
# 5) 화면에 FPS / 추론 시간 / 마지막 예측 경과 시간 표시

import time
import threading
from pathlib import Path

import cv2
import numpy as np
import torch
import main_script


# =========================================================
# 1. 설정
# =========================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "best_epoch012_val0.6905.pt"

# CUDA가 있으면 사용하고, 없으면 CPU 사용
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CAMERA_INDEX = 0

# 먼저 640x480으로 테스트. 여전히 끊기면 320x240으로 낮춰볼 것.
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480

# CPU에서 AI 추론이 화면 표시를 방해하지 않도록 스레드 수 제한
CPU_THREADS = 2

# 모델 추론 간격. 0.40초 = 최대 약 2.5회/초
# 화면 프레임은 이 간격과 무관하게 계속 표시됨.
INFERENCE_INTERVAL = 0.40

# EMA 스무딩: 낮을수록 예측값은 부드럽지만 반응은 느려짐
SMOOTHING_ALPHA = 0.50

# 기존 코드에서 사용한 표준화 통계값
VEC_MEAN = np.array(
    [51.0059475, 9094.12979167, 31725.918125], dtype=np.float32
)
VEC_STD = np.array(
    [11.65144146, 636.66968797, 1533.04362754], dtype=np.float32
)

# 기존 코드에서 사용한 데이터 범위 기반 정규화 상수
ROUGHNESS_MIN, ROUGHNESS_MAX = 25.3844656, 77.31173003
HARDNESS_MIN, HARDNESS_MAX = 7188.6875, 10151.125
FRICTION_MIN, FRICTION_MAX = 28223.1875, 33878.59375


def configure_torch():
    """모델 연산을 시작하기 전에 CPU 스레드 수를 설정한다."""
    if DEVICE.type == "cpu":
        torch.set_num_threads(CPU_THREADS)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # 이미 PyTorch 연산이 시작된 경우 설정이 늦었을 수 있음
            pass


def normalize_minus1_to_1(value, value_min, value_max):
    if value_max <= value_min:
        raise ValueError("정규화 최대값은 최소값보다 커야 합니다.")
    normalized = 2.0 * (float(value) - value_min) / (value_max - value_min) - 1.0
    return float(np.clip(normalized, -1.0, 1.0))


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"모델 파일을 찾을 수 없습니다: {MODEL_PATH}\n"
            "best_epoch012_val0.6905.pt를 이 파일과 같은 폴더에 두세요."
        )

    configure_torch()

    model = main_script.build_model(
        arch="fcn_resnet50",
        mode="regression",
        pretrained=False,
        dec_ch=128
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )
    model.load_state_dict(checkpoint["model"])
    model.to(DEVICE)
    model.eval()

    print(f"모델 로딩 완료 / 실행 장치: {DEVICE}")
    print("Epoch:", checkpoint.get("epoch", "정보 없음"))
    print("Validation Loss:", checkpoint.get("val_loss", "정보 없음"))
    return model


def preprocess_frame(frame):
    """화면 중앙의 작은 ROI(가로 40%, 세로 50%)를 224x224로 바꾸고 /255 처리."""
    height, width = frame.shape[:2]
    roi_width = max(1, int(width * 0.40))
    roi_height = max(1, int(height * 0.50))
    x1 = (width - roi_width) // 2
    y1 = (height - roi_height) // 2
    x2, y2 = x1 + roi_width, y1 + roi_height

    roi = frame[y1:y2, x1:x2]
    rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (224, 224), interpolation=cv2.INTER_LINEAR)

    tensor = (
        torch.from_numpy(resized.copy())
        .permute(2, 0, 1)
        .float()
        .div_(255.0)
        .unsqueeze(0)
    )
    return tensor, (x1, y1, x2, y2)


def predict_frame(frame, model):
    tensor, roi_box = preprocess_frame(frame)
    tensor = tensor.to(DEVICE, non_blocking=(DEVICE.type == "cuda"))

    with torch.inference_mode():
        output = model(tensor)

    prediction_map = output["vec"][0].detach().float().cpu().numpy()
    standardized = np.median(prediction_map.reshape(3, -1), axis=1)
    raw = standardized * VEC_STD + VEC_MEAN

    result = {
        "roughness": normalize_minus1_to_1(
            raw[0], ROUGHNESS_MIN, ROUGHNESS_MAX
        ),
        "hardness": normalize_minus1_to_1(
            raw[1], HARDNESS_MIN, HARDNESS_MAX
        ),
        "friction": normalize_minus1_to_1(
            raw[2], FRICTION_MIN, FRICTION_MAX
        )
    }
    return result, roi_box


class CameraWorker:
    """카메라에서 프레임을 읽는 역할만 수행한다."""

    def __init__(self):
        self.cap = None
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None
        self.frame = None
        self.frame_id = 0
        self.error = None

    def start(self):
        # Windows에서는 DirectShow를 우선 사용
        self.cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = cv2.VideoCapture(CAMERA_INDEX)

        if not self.cap.isOpened():
            raise RuntimeError(
                "웹캠을 열 수 없습니다. 연결 상태와 CAMERA_INDEX를 확인하세요."
            )

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
        # 지원 여부는 드라이버/백엔드에 따라 다름
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        actual_w = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_h = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        backend = self.cap.getBackendName()
        print(
            f"카메라: {actual_w:.0f}x{actual_h:.0f}, "
            f"보고 FPS={fps:.1f}, 백엔드={backend}"
        )

        self.thread = threading.Thread(
            target=self._run, name="CameraWorker", daemon=True
        )
        self.thread.start()
        return self

    def _run(self):
        while not self.stop_event.is_set():
            ok, frame = self.cap.read()
            if not ok:
                self.error = "웹캠 프레임 읽기 실패"
                time.sleep(0.02)
                continue

            with self.lock:
                self.frame = frame
                self.frame_id += 1
                self.error = None

    def get_latest(self):
        with self.lock:
            if self.frame is None:
                return None, self.frame_id
            return self.frame.copy(), self.frame_id

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=1.0)
        if self.cap is not None:
            self.cap.release()
            self.cap = None


class InferenceWorker:
    """AI 추론만 수행한다. 화면 표시 루프와 별도 스레드다."""

    def __init__(self, camera, model):
        self.camera = camera
        self.model = model
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread = None
        self.result = {
            "roughness": None,
            "hardness": None,
            "friction": None,
            "valid": False,
            "timestamp": 0.0,
            "inference_ms": 0.0
        }
        self.error = None

    def start(self):
        self.thread = threading.Thread(
            target=self._run, name="InferenceWorker", daemon=True
        )
        self.thread.start()
        return self

    def _run(self):
        smoothed = None
        last_inference = 0.0
        last_frame_id = -1

        while not self.stop_event.is_set():
            now = time.monotonic()
            if now - last_inference < INFERENCE_INTERVAL:
                time.sleep(0.005)
                continue

            frame, frame_id = self.camera.get_latest()
            if frame is None or frame_id == last_frame_id:
                time.sleep(0.01)
                continue

            last_frame_id = frame_id
            last_inference = time.monotonic()
            start_time = time.perf_counter()

            try:
                current, _ = predict_frame(frame, self.model)
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0

                if smoothed is None:
                    smoothed = current.copy()
                else:
                    for key in ("roughness", "hardness", "friction"):
                        smoothed[key] = (
                            SMOOTHING_ALPHA * current[key]
                            + (1.0 - SMOOTHING_ALPHA) * smoothed[key]
                        )

                new_result = {
                    "roughness": float(smoothed["roughness"]),
                    "hardness": float(smoothed["hardness"]),
                    "friction": float(smoothed["friction"]),
                    "valid": True,
                    "timestamp": time.monotonic(),
                    "inference_ms": elapsed_ms
                }
                with self.lock:
                    self.result = new_result
                    self.error = None

            except Exception as exc:
                with self.lock:
                    self.error = repr(exc)
                time.sleep(0.05)

    def get_latest(self):
        with self.lock:
            return dict(self.result)

    def get_error(self):
        with self.lock:
            return self.error

    def stop(self):
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)


def main():
    model = load_model()
    camera = CameraWorker().start()
    inference = InferenceWorker(camera, model).start()

    display_count = 0
    fps_start = time.monotonic()
    display_fps = 0.0

    try:
        while True:
            # 이 루프에서는 모델 추론을 절대 실행하지 않는다.
            frame, _ = camera.get_latest()
            if frame is None:
                time.sleep(0.005)
                continue

            result = inference.get_latest()
            error = inference.get_error()

            display_count += 1
            now = time.monotonic()
            if now - fps_start >= 1.0:
                display_fps = display_count / (now - fps_start)
                display_count = 0
                fps_start = now

            h, w = frame.shape[:2]
            roi_w = int(w * 0.40)
            roi_h = int(h * 0.50)
            x1 = (w - roi_w) // 2
            y1 = (h - roi_h) // 2
            cv2.rectangle(
                frame, (x1, y1), (x1 + roi_w, y1 + roi_h),
                (0, 255, 0), 2
            )

            lines = [f"Display FPS: {display_fps:.1f}"]
            if result["valid"]:
                age = max(0.0, now - result["timestamp"])
                lines += [
                    f"Roughness: {result['roughness']:.3f}",
                    f"Hardness:  {result['hardness']:.3f}",
                    f"Friction:  {result['friction']:.3f}",
                    f"Inference: {result['inference_ms']:.0f} ms",
                    f"Result age: {age:.2f} s"
                ]
            else:
                lines.append("Waiting for first prediction...")

            if error:
                lines.append("Inference error (see terminal)")

            for i, line in enumerate(lines):
                cv2.putText(
                    frame, line, (15, 28 + i * 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58,
                    (0, 255, 0) if i == 0 else (255, 255, 255), 2
                )

            cv2.putText(
                frame, "Q: quit", (15, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1
            )
            cv2.imshow("Material Prediction - Decoupled", frame)

            # waitKey는 GUI 이벤트 처리에 필요
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # 결과가 너무 오래됐으면 콘솔에서 원인을 확인하기 쉽도록 표시
            if result["valid"] and now - result["timestamp"] > 2.0:
                pass

    finally:
        inference.stop()
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
