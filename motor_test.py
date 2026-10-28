# motor_test.py — RoboKit RS / A0 button board / D12,D13 positional servos
# Webcam material prediction + button-driven, ABSOLUTE servo target positions
# Existing RobokitRS import and constructor kept as requested.

import time
import importlib.util
from pathlib import Path

import cv2
from RobokitRS import *

ROOT = Path(__file__).resolve().parent
COM_PORT = "COM5"
BUTTON_ANALOG_PORT = 0
SERVO_D12 = 12
SERVO_D13 = 13

# A0 readings measured on this robot
WHITE_MIN, WHITE_MAX = 380, 440  # white -> grasp
RED_MIN, RED_MAX = 280, 340      # red   -> open
BUTTON_STABLE_READS = 2
BUTTON_POLL_SECONDS = 0.10
MIN_ACTION_GAP = 0.65
MAX_PREDICTION_AGE = 2.0
SEND_GAP = 0.15

# =====================================================================
# [IMPORTANT 1] 집게가 "일자로 벌어진" 상태의 두 모터 절대 위치를 조절하는 곳
# =====================================================================
# 이전 값은 90/90이었음. 이번 초깃값은 75/95로 변경해 놓았지만,
# 사진만으로 '일자'가 되는 정확한 위치는 알 수 없음.
# 먼저 링크의 간섭이 없는 상태에서 O 키로 확인하고 A/Z, S/X로 미세 조절.
# 실제 일자 위치를 찾으면 여기의 숫자를 확정해 적어 둘 것.
OPEN_D12 = 75
OPEN_D13 = 95

# =====================================================================
# [IMPORTANT 2] 파지 수준에 따라 모터가 이동할 "절대 목표 위치"를 조절
# =====================================================================
# 점수가 낮으면 SOFT 위치, 높으면 STRONG 위치. 현재 위치에서 더하지 않음!
# 예) 0.5라면 SOFT와 STRONG의 정확히 중간 절대 각도로 이동함.
# 본 값은 물리적 파지력(N)이 아니라 서보의 위치(각도 명령)임.
GRIP_D12_SOFT = 81
GRIP_D12_STRONG = 89
GRIP_D13_SOFT = 101
GRIP_D13_STRONG = 109

# =====================================================================
# [IMPORTANT 3] 두 모터의 "실제 회전 방향"을 조절하는 곳
# =====================================================================
# 이전 코드에서 D12는 +, D13은 -였는데 실제로 같은 방향으로 돌았다고 하였음.
# 그래서 이번에는 D12/D13의 목표 위치 숫자를 둘 다 증가시키도록 변경함.
# 두 축이 서로 반대 방향으로 돌아야 정상. 실제로 같게 돌면
# D13은 OPEN_D13보다 작은 위치를 향하게 SOFT/STRONG값을 수정할 것.
# 예: GRIP_D13_SOFT=89, GRIP_D13_STRONG=81 (기계적 간섭 확인 필수)

# Limits are conservative provisional checks, not verified mechanical endpoints.
SERVO_LIMIT_MIN = 60
SERVO_LIMIT_MAX = 120
FINE_STEP = 2

# Safety: start locked; pressing E intentionally activates and sends OPEN targets.
# Press O to request OPEN position again; Q quits. No automatic grasp on startup.
MOTOR_ARMED = False


def load_prediction_module():
    for filename in ("model_test(2).py", "model_test.py"):
        filepath = ROOT / filename
        if not filepath.is_file():
            continue
        spec = importlib.util.spec_from_file_location("gripper_model_test", filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        required = ("load_model", "CameraWorker", "InferenceWorker")
        if all(hasattr(module, attr) for attr in required):
            print("재질 예측 모듈:", filename)
            return module
    raise RuntimeError("CameraWorker/InferenceWorker가 있는 model_test.py 또는 model_test(2).py가 필요합니다.")


def classify_button(raw):
    if raw is None:
        return "UNKNOWN"
    if WHITE_MIN <= raw <= WHITE_MAX:
        return "WHITE"
    if RED_MIN <= raw <= RED_MAX:
        return "RED"
    return "RELEASED"


def choose_grip_level(hardness, friction):
    """[-1,1] 정규화 예측값 -> [0,1] 실험용 파지 수준. 실제 힘(N)이 아님."""
    hardness_level = max(0.0, min(1.0, (float(hardness) + 1.0) / 2.0))
    low_friction_level = max(0.0, min(1.0, (1.0 - float(friction)) / 2.0))
    return 0.4 * hardness_level + 0.6 * low_friction_level


def choose_grip_positions(hardness, friction):
    """항상 같은 예측값에 같은 절대 위치를 대응시킨다(누적 이동 없음)."""
    level = choose_grip_level(hardness, friction)
    p12 = round(GRIP_D12_SOFT + level * (GRIP_D12_STRONG - GRIP_D12_SOFT))
    p13 = round(GRIP_D13_SOFT + level * (GRIP_D13_STRONG - GRIP_D13_SOFT))
    return int(p12), int(p13), level


def validate_positions(p12, p13):
    for name, pos in (("D12", p12), ("D13", p13)):
        if not SERVO_LIMIT_MIN <= pos <= SERVO_LIMIT_MAX:
            raise ValueError(f"{name} 위치 {pos} 제한 범위 초과: {SERVO_LIMIT_MIN}~{SERVO_LIMIT_MAX}")


def configure_servo_pins(robot):
    modes = getattr(RobokitRS, "Modes", None)
    if modes is None:
        modes = getattr(type(robot), "Modes", None)
    if modes is None or not hasattr(modes, "SERVO"):
        raise RuntimeError("RobokitRS.Modes.SERVO를 찾지 못했습니다. 패키지 버전을 확인하세요.")
    robot.set_pin_mode(SERVO_D12, modes.SERVO)
    time.sleep(0.12)
    robot.set_pin_mode(SERVO_D13, modes.SERVO)
    time.sleep(0.20)
    print("D12/D13 서보 모드 설정 명령 전송")


def send_servos(robot, p12, p13, label):
    validate_positions(p12, p13)
    print(f"[{label}] 목표 절대 위치 D12={p12}, D13={p13} / ARMED={MOTOR_ARMED}", flush=True)
    if not MOTOR_ARMED:
        print("[모터 잠금] 실제 명령은 전송하지 않았습니다. E를 눌러 활성화하세요.")
        return False
    robot.servo_write(SERVO_D12, int(p12))
    time.sleep(SEND_GAP)
    robot.servo_write(SERVO_D13, int(p13))
    print("servo_write 명령 전송 완료 (실제 이동 확인 필요)", flush=True)
    return True


def main():
    global MOTOR_ARMED

    model_test = load_prediction_module()
    print("AI 모델 로딩 중...")
    model = model_test.load_model()
    camera = model_test.CameraWorker().start()
    inference = model_test.InferenceWorker(camera, model).start()
    robot = None

    open12, open13 = OPEN_D12, OPEN_D13
    last_raw = None
    button = "UNKNOWN"
    last_candidate = "UNKNOWN"
    stable_count = 0
    stable_button = "UNKNOWN"
    released_since_press = False
    last_action = 0.0
    last_poll = 0.0
    message = "E unlock + OPEN | O open | T target test | Q quit"

    try:
        # Requested RobokitRS import and construction syntax retained exactly.
        robot = RobokitRS.RobokitRS()
        robot.port_open(COM_PORT)
        time.sleep(0.7)
        configure_servo_pins(robot)

        print(f"버튼: WHITE={WHITE_MIN}~{WHITE_MAX}, RED={RED_MIN}~{RED_MAX} (A0)")
        print(f"[OPEN 초기 각도] D12={open12}, D13={open13}")
        print("E=활성화/잠금(+활성화시 열림 목표 전송), O=열기, T=약한 파지 목표 테스트, Q=종료")
        print("A/Z=D12 열림위치 ±2, S/X=D13 열림위치 ±2 (임시보정)")

        while True:
            now = time.monotonic()
            result = inference.get_latest()
            if now - last_poll >= BUTTON_POLL_SECONDS:
                last_poll = now
                try:
                    adc = robot.analog_read(BUTTON_ANALOG_PORT)
                    last_raw = None if adc is None else int(adc)
                except Exception as exc:
                    last_raw = None
                    message = f"A0 error: {exc}"
                    print(message)

                button = classify_button(last_raw)
                if button == last_candidate:
                    stable_count += 1
                else:
                    last_candidate = button
                    stable_count = 1

                if stable_count >= BUTTON_STABLE_READS and button != stable_button:
                    stable_button = button
                    if button == "RELEASED":
                        released_since_press = True
                        message = "Button released - ready"
                    elif button in ("WHITE", "RED") and released_since_press:
                        released_since_press = False
                        if now - last_action >= MIN_ACTION_GAP:
                            last_action = now
                            if button == "WHITE":
                                age = now - result.get("timestamp", 0.0)
                                if not result.get("valid") or age > MAX_PREDICTION_AGE:
                                    message = f"WHITE: AI result missing / old ({age:.1f}s)"
                                    print(message)
                                else:
                                    p12, p13, level = choose_grip_positions(
                                        result["hardness"], result["friction"]
                                    )
                                    message = f"WHITE grip level={level:.2f}: D12={p12}, D13={p13}"
                                    send_servos(robot, p12, p13, "WHITE GRIP")
                            else:
                                message = f"RED OPEN: D12={open12}, D13={open13}"
                                send_servos(robot, open12, open13, "RED OPEN")

            frame, _ = camera.get_latest()
            if frame is not None:
                h, w = frame.shape[:2]
                rw, rh = int(w * 0.40), int(h * 0.50)
                x1, y1 = (w - rw) // 2, (h - rh) // 2
                cv2.rectangle(frame, (x1, y1), (x1 + rw, y1 + rh), (0, 255, 0), 2)
                lines = [
                    f"A0={last_raw} / button={button}",
                    f"WHITE {WHITE_MIN}-{WHITE_MAX} / RED {RED_MIN}-{RED_MAX}",
                    f"ARMED={MOTOR_ARMED} | OPEN D12={open12}, D13={open13}",
                    f"SOFT D12={GRIP_D12_SOFT}, D13={GRIP_D13_SOFT}",
                    f"STRONG D12={GRIP_D12_STRONG}, D13={GRIP_D13_STRONG}",
                ]
                if result.get("valid"):
                    p12, p13, level = choose_grip_positions(result["hardness"], result["friction"])
                    lines.extend([
                        f"Hardness={result['hardness']:.3f} Friction={result['friction']:.3f}",
                        f"Grip level={level:.2f} target D12={p12}, D13={p13}",
                        f"Inference={result.get('inference_ms', 0):.0f} ms",
                    ])
                else:
                    lines.append("Waiting for AI prediction...")
                lines.extend([message[:75], "E arm/open | O open | T soft grip | A/Z D12 | S/X D13 | Q quit"])
                for i, line in enumerate(lines):
                    cv2.putText(frame, line, (8, 24 + i * 26),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                cv2.imshow("RS Gripper - absolute positions", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord("e"):
                MOTOR_ARMED = not MOTOR_ARMED
                message = f"ARMED={MOTOR_ARMED}"
                print(message)
                if MOTOR_ARMED:
                    # Initial OPEN position sent only after user intentionally arms.
                    send_servos(robot, open12, open13, "INITIAL OPEN")
            elif key == ord("o"):
                message = f"KEY OPEN D12={open12} D13={open13}"
                send_servos(robot, open12, open13, "KEY OPEN")
            elif key == ord("t"):
                # This is an absolute target, not 'add a bit more to previous position'.
                message = f"KEY SOFT TARGET D12={GRIP_D12_SOFT} D13={GRIP_D13_SOFT}"
                send_servos(robot, GRIP_D12_SOFT, GRIP_D13_SOFT, "KEY SOFT TEST")
            elif key in (ord("a"), ord("z"), ord("s"), ord("x")):
                d12 = FINE_STEP if key == ord("a") else (-FINE_STEP if key == ord("z") else 0)
                d13 = FINE_STEP if key == ord("s") else (-FINE_STEP if key == ord("x") else 0)
                new12, new13 = open12 + d12, open13 + d13
                try:
                    validate_positions(new12, new13)
                    open12, open13 = new12, new13
                    message = f"OPEN fine-adjust D12={open12}, D13={open13}"
                    print(message)
                    send_servos(robot, open12, open13, "OPEN fine-adjust")
                except ValueError as exc:
                    message = str(exc)
                    print(message)

            time.sleep(0.008)
    finally:
        MOTOR_ARMED = False
        if robot is not None:
            try:
                robot.end()
            except Exception as exc:
                print("RobokitRS end 오류:", exc)
        inference.stop()
        camera.stop()
        cv2.destroyAllWindows()
        print("종료")


if __name__ == "__main__":
    main()