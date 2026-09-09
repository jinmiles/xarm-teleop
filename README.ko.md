# xarm-teleop

> English: [README.md](README.md)

**UFACTORY xArm7 + Inspire RH56 5지 핸드**의 실시간 텔레오퍼레이션. Intel RealSense **D435**를
손에 향하게 두면 로봇 엔드이펙터가 손목을 따라오고, RH56의 각 손가락이 대응하는 손가락을 따라옵니다.

손이 **어디에** 있는지는 항상 카메라가 제공합니다. 손이 **어떤 자세인지**는 두 가지 소스 중 하나에서
오며, 실행 시 `--pose-source`로 고릅니다.

```
POSITION  D435 color+depth ─► WiLoR 손목 픽셀 + metric depth ─► wrist → TCP ─► safety ─► xArm7

POSE      1번 방법 (기본)   WiLoR 21 keypoints, 마커리스   ─┐
          2번 방법          Noitom 모캡 장갑 (랜선)        ─┴─► fingers → RH56 6 DOF (RS485)
```

**1번 방법**은 카메라만 있으면 됩니다. **2번 방법**은 IMU로 손가락 굴곡을 직접 재는 모캡 장갑을
더합니다. 그래서 카메라가 못 보는 상황 — 물체를 감싸 쥔 주먹, 손바닥 뒤로 넘어간 손가락, 모션
블러 — 에서도 손가락이 계속 동작합니다. 대신 장갑을 착용해야 하고, 이 PC와 랜선으로 연결된 윈도우
노트북에서 Axis Studio를 띄워야 합니다.

## 1. 동작 원리

**손목 → 팔.** WiLoR는 스케일이 확정되지 않은 손목 포즈를 줍니다. 손목 픽셀의 D435 depth가
신뢰할 수 없는 카메라 Z를 대체하므로 이동량이 metric이 됩니다(`--scale 1 --depth-scale 1`로 실행).
제어는 **상대(클러치)** 방식입니다. 첫 추적 프레임에서 engage 하고 그 지점부터 손 움직임을 따라가므로,
추적을 일부러 놓았다가 다시 잡으면 재인덱싱이 됩니다. 모든 타깃은 워크스페이스 박스와 틱당 최대
이동량으로 클램프된 뒤에 팔에 전달됩니다.

**손가락 → 핸드.** 손가락별 curl은 각 손가락 체인을 따라 굴곡각을 합한 값이고, 엄지 회전은 엄지
중수골이 손바닥 평면에서 벗어난 각도입니다. 둘 다 **각도**이므로 손 크기와 무관합니다. 다만 사람마다
가동범위는 다르고, 그걸 잡아주는 것이 §3의 캘리브레이션입니다. 결과는 `[0,1]` 범위의 닫힘 비율 6개이며
핸드의 6 DOF에 매핑됩니다.

**장갑 (2번 방법).** 윈도우 노트북의 Axis Studio가 장갑의 본 회전을 랜선으로 브로드캐스트하면,
이 PC가 받아서 손 스켈레톤에 대해 forward kinematics를 돌려 **WiLoR와 동일한 21 키포인트 형식**을
만듭니다. 그 뒤 단계 — curl, 엄지 회전, 캘리브레이션, 오버레이 — 는 위에 설명한 코드 그대로이며 분기가
없습니다. 장갑은 IMU 장치라 위치는 절대 주지 않는데, 그 부분이 정확히 카메라가 담당하는 절반입니다.

두 센서는 서로 다른 좌표계에 있으므로, 그 사이 회전은 **설정하지 않고 측정**합니다. 두 센서가 같은
손바닥을 보므로, 양쪽 키포인트에서 같은 방식으로 만든 palm frame의 차이가 곧 그 회전입니다. 매 재획득
직후 `--glove-align-frames` 프레임만큼 평균 내어 고정하므로, 클러치를 재인덱싱하면 장갑 정합도 함께
다시 잡힙니다. 손가락 각도는 이 정합과 무관하므로 첫 프레임부터 정상 동작합니다.

**RH56 링크.** 이 핸드는 벤더 매뉴얼의 `EB 90` 프레이밍이 아니라 **Modbus RTU**(slave id 1,
8N1 115200)로 말합니다. 레지스터 `1040..1045`에 big-endian int16 6개,
순서는 `[little, ring, middle, index, thumb_bend, thumb_rot]`이며, 값은 매뉴얼의 0–1000 스케일이
아니라 장치 원시 단위입니다. 드라이버가 하드웨어로 검증된 두 포즈 사이를 보간하고 그 범위로 클램프하므로
DOF별 방향은 자동 처리됩니다 — 네 손가락과 엄지 굽힘은 값이 **줄어들며** 닫히고, 엄지 회전은
**늘어나며** 대립합니다.

```python
# src/control/inspire_hand.py
CMD_OPEN   = [1740, 1740, 1740, 1740, 1350, 1500]
CMD_CLOSED = [1400, 1400, 1400, 1400, 1250, 1650]   # 완전한 주먹이 아니라 가벼운 그립
```

`CMD_CLOSED`는 의도적으로 실제 엔드스톱보다 안쪽입니다. `hand-test`로 엔드스톱을 확인한 뒤에 넓히세요.

동일한 코드가 하나의 백엔드 인터페이스를 통해 **MuJoCo 시뮬레이션**과 실기를 모두 구동하므로, 실기를
움직이기 전에 시뮬레이션 팔로 손가락 리타게팅을 검증할 수 있습니다.

## 2. 설치

**요구 사항** — Linux, Python **3.13**, conda, CUDA 11.8 지원 드라이버가 설치된 NVIDIA GPU
(~4 GB VRAM), USB 3.0에 연결된 RealSense **D435**, 네트워크에 연결된 **xArm7**, RS485로 연결된
Inspire **RH56**(전용 USB 어댑터, 예: `/dev/ttyUSB0`).

```bash
bash scripts/install_env.sh          # env 이름: xarm-teleop (인자로 변경 가능)
conda activate xarm-teleop
python scripts/teleop.py wilor-image # 모델 확인; 가중치는 HF 캐시로 다운로드(~2 GB)
```

이 스크립트는 cu118 인덱스에서 torch를 먼저 설치하고, chumpy·opencv보다 앞서 numpy를 고정한 뒤,
chumpy를 `--no-build-isolation`으로 빌드하고, WiLoR-mini를 `--no-deps`로 설치합니다. 새 머신에서는
이 순서가 중요합니다 — 직접 단계별로 돌리고 싶다면 스크립트를 보세요. 시뮬레이션 백엔드를 쓰려면
xArm7 MuJoCo 모델을 한 번 받아둡니다.

```bash
git clone --depth 1 --filter=blob:none \
    https://github.com/google-deepmind/mujoco_menagerie.git third_party/mujoco_menagerie
```

**하드웨어 점검**

```bash
rs-enumerate-devices | head        # D435가 USB 3.0(파란색) 포트에 있는지
ping <YOUR_XARM_IP>                # xArm7; UFACTORY Studio에서 원격 모션 허용, 에러 없음
ls -l /dev/ttyUSB0                 # RH56; 접근하려면 dialout 그룹에 사용자 추가
```

그다음 핸드만 따로 올려봅니다. 카메라나 팔이 개입하기 전에 RS485 링크를 확인하는 가장 빠른 방법입니다.

```bash
python scripts/teleop.py hand-test --port /dev/ttyUSB0
```

핸드가 Modbus 읽기에 응답하지 않으면 시작을 거부하고, 응답하면 각 DOF를 열림→굽힘→열림으로 훑습니다.
이름이 가리키는 손가락이 실제로 움직이는지 확인하세요.

### 2.1 장갑과 윈도우 노트북 (2번 방법 전용)

Noitom 장갑과 허브, Axis Studio 라이선스가 있는 윈도우 노트북, 그리고 그 노트북에서 텔레옵 PC까지
닿는 네트워크 경로가 필요합니다. 1번 방법만 쓸 거면 이 절 전체를 건너뛰세요.

**a. 두 대를 네트워크에 올리기.** 같은 서브넷일 필요는 없고, 서로 닿기만 하면 됩니다.
**Destination은 `teleop.py`를 돌릴 PC**입니다. 장갑 스트림이 그리로 가므로, 그 머신에 이 repo,
conda 환경, D435, RH56이 있어야 합니다.

| Axis Studio 항목 | 머신 | 이 장비 기준 | 역할 |
|---|---|---|---|
| *Local Address* | 윈도우 노트북 (Wi-Fi) | `192.168.0.46`, 포트 `7001` | Axis Studio 실행, 송신 |
| *Destination Address* | 우분투 텔레옵 PC | `10.20.26.115`, 포트 `7012` | `teleop.py` 실행, 수신 |

주소를 직접 확인하려면:

```cmd
:: 윈도우 — Wi-Fi 어댑터의 IPv4, 그리고 목적지로 실제 나가는 주소
ipconfig | findstr /i "IPv4"
powershell -c "Find-NetRoute -RemoteIPAddress 10.20.26.115"    :: IPAddress 값이 Local Address
```

```bash
# 텔레옵 PC — 자기 LAN 주소
hostname -I | awk '{print $1}'
ip -4 -br addr show scope global | grep -vE "docker|br-|virbr|tailscale"
```

`127.0.0.1`, 가상 어댑터(`vEthernet`, VirtualBox, `docker0`, `br-*`, `virbr0`), Tailscale `100.x`는
제외하세요 — 상대 머신이 닿을 수 있는 주소가 아닙니다. 그다음 경로가 살아 있는지 확인:
윈도우에서 `ping 10.20.26.115`. 윈도우 방화벽 창이 뜨면 **개인 네트워크에서 허용**을 선택합니다.

Wi-Fi로도 스트림은 잘 옵니다. 다만 링크가 혼잡하면 `glove frames are older than 0.30s` 경고와 함께
프레임이 hold 됩니다. 그러면 노트북을 랜선으로 옮기거나 `--glove-timeout`을 올리세요. 두 서브넷 사이에서
UDP가 막히는 경우(§6)에는 랜선 직결로 우회합니다 — 윈도우에 고정 `192.168.2.16`,
텔레옵 PC에 `sudo ip addr add 192.168.2.15/24 dev <iface>`를 주고 그 두 주소를 쓰면 됩니다.

**b. Axis Studio.** 라이선스 동글을 꽂은 상태로 윈도우 노트북에 설치하고, 장갑 허브 전원을 켜고 장갑을
페어링한 뒤, **Settings → Working Mode**를 **Hand** 모드로 둡니다(그래야 스트림에 손가락 본이
실립니다). 그다음 Axis Studio의 안내 포즈로 장갑을 캘리브레이션합니다. §3의 텔레옵 캘리브레이션은
이것과 별개인 이후 단계이며, 이 과정을 대체하지 않습니다.

**c. BVH Broadcasting.** **Settings → BVH Broadcasting**을 열고 우측 상단 토글을 켠 뒤, 아래 값을
그대로 설정합니다. 하나라도 다르면 수신 측이 아무것도 못 받거나, 읽을 수 없는 스켈레톤을 받습니다.

| 항목 | 값 |
|---|---|
| Frame Format → Type | `Binary`, *Use old header format* 체크 해제 |
| Sync | `GenLock` 체크 |
| Skeleton | `Axis Studio` |
| BVH Format → Rotation | `XYZ` |
| BVH Format → Displacement | **체크** — 본 길이가 이 옵션으로 실려 옵니다 |
| Coordinate system | `OPT` |
| Protocol | `UDP` |
| Local Address | **a**에서 확인한 윈도우 주소(`192.168.0.46`), 포트 `7001` |
| Destination Address | **텔레옵 PC**의 주소(`10.20.26.115`), 포트 `7012` |

루프백 주소(`127.0.0.1`)는 절대 동작하지 않습니다. Destination은 텔레옵 PC가 `ping`에 응답하는 그
주소여야 합니다. OK를 누르면 즉시 송출이 시작되고 Axis Studio가 켜져 있는 동안 유지됩니다.
*Destination Address* 옆 `+` 버튼으로 수신처를 더 추가할 수 있어서, 장갑 하나로 개발용 PC와 로봇 PC에
동시에 보낼 수 있습니다(둘 다 포트 `7012`).

*Displacement*가 사람들이 가장 자주 빠뜨리는 설정입니다. 끄면 회전은 오지만 본 길이가 오지 않고,
수신 측이 표준 손 스켈레톤으로 폴백합니다(경고 로그로 알려줍니다). 그러면 손가락 각도가 근사값이
됩니다.

*UDP 대신 TCP*: 같은 창에서 Protocol을 `TCP`로 두고
`--glove-host 192.168.2.16 --glove-port <해당 포트>`로 실행하면, 이 PC가 브로드캐스트를 기다리는
대신 Axis Studio에 접속합니다.

**d. 링크부터 확인하세요** — 카메라·팔·핸드가 개입하기 전에. 두 명령 모두 Axis Studio가 송출 중인
상태에서, **텔레옵 PC**에서 실행합니다:

```bash
sudo tcpdump -i any -n udp port 7012             # 패킷이 오기는 하는지; Ctrl+C로 종료
python scripts/teleop.py glove-test              # 파이프라인이 읽는지; Ctrl+C로 종료
```

`tcpdump`가 네트워크 문제와 파싱 문제를 갈라줍니다. 여기가 조용하면 주소나 방화벽 문제(§6),
여기는 찍히는데 `glove-test`가 조용하면 워킹 모드나 손 선택 문제입니다.

프레임이 실제로 도착하지 않으면 시작을 거부하고, 도착하면 6개 DOF의 raw 각도(캘리브레이션이 있으면
0–1 비율도 함께)를 초당 몇 번씩 찍습니다. 손가락을 하나씩 굽히면서 해당 숫자만 움직이는지 보세요.
반대 손에 장갑을 꼈다면 `--glove-hand left`, Destination 포트를 바꿨다면 `--glove-port <n>`을
붙입니다.

수신부는 `mocap_ros_py`의 Noitom MocapApi 래퍼를 `/home/user/extra_workdir/mocap_ros_py`에서 그대로
가져다 씁니다(`src/paths.py` 참고). 양쪽 어디에도 ROS는 쓰이지 않습니다.

## 3. 캘리브레이션

조작자마다 한 번 실행합니다. 본인의 편 손과 주먹 각도를 기록해서, raw curl을 쓸 수 있는 0–1 비율로
바꿔주는 단계입니다.

```bash
python scripts/teleop.py hand-calib --source realsense                  # 1번 방법
python scripts/teleop.py hand-calib --pose-source glove                 # 2번 방법 (카메라 불필요)
```

소스마다 파일이 분리됩니다 — 카메라는 `data/hand_calib.json`, 장갑은 `data/glove_hand_calib.json`.
같은 손이라도 두 센서의 측정값이 다르기 때문입니다. `sim`/`teleop`이 알아서 맞는 파일을 집어 오고,
`--hand-calib <path>`로 다른 경로를 지정하거나 `--hand-calib none`으로 내장 기본값을 강제할 수
있습니다. 장갑 캡처는 카메라·팔·핸드가 전혀 필요 없습니다. 관절 각도만 기록하는데, 그건 장갑이 단독으로
제공하기 때문입니다.
**카운트다운이 끝나기 전에 포즈를 취하고 있어야 합니다** — "open" 캡처가 사실 반쯤 쥔 주먹이면 쓸 만한
구간이 남지 않고, 모든 DOF가 상수로 포화합니다. 시작 시 구간 폭을 로그로 찍고, 0.15 rad 미만인 DOF는
경고합니다.

## 4. 5지 텔레옵 실행

먼저 시뮬레이션 팔로 검증합니다 — 핸드는 실물, 팔은 가상:

```bash
python scripts/teleop.py sim --source realsense --scale 1.0 --depth-scale 1.0 \
    --hand-port /dev/ttyUSB0 --display
```

손가락을 하나씩 굽혀 의도한 손가락이 움직이는지 확인하세요. 창에 DOF마다 막대가 하나씩 표시되어 무엇이
명령되는지 볼 수 있습니다. 그다음 실기를, 저속으로 E-stop을 손 닿는 곳에 두고:

```bash
python scripts/teleop.py teleop --execute --ip <YOUR_XARM_IP> \
    --source realsense --scale 1.0 --depth-scale 1.0 --tcp-speed 80 \
    --hand-port /dev/ttyUSB0 --display
```

`--execute`를 빼면 팔에 접속하지 않고 모든 명령만 생성하는 dry-run이 됩니다. 헤드리스 환경에서는
`--display`를 빼세요. `outputs/`로의 녹화에는 영향이 없습니다.

**2번 방법**은 같은 명령에 `--pose-source glove`만 붙이면 됩니다. 손목 위치는 여전히 카메라가
담당하므로 `--source`는 그대로 필요합니다.

```bash
python scripts/teleop.py sim --source realsense --scale 1.0 --depth-scale 1.0 \
    --pose-source glove --hand-port /dev/ttyUSB0 --display
```

장갑이 프레임을 보내기 전에는 실행을 거부하고, 오버레이에는 `pose:glove` 라벨이 붙으며, 화면에 그려지는
손 스켈레톤은 장갑의 자세를 사용자 손 위에 재투영한 것입니다 — 즉 영상이 RH56에 실제로 전달되는 자세를
보여줍니다. 장갑이 `--glove-timeout`보다 오래 조용하면 그 프레임은 미추적으로 처리됩니다. 카메라가 손을
놓쳤을 때와 똑같이 팔은 hold하고 손가락은 마지막 자세를 유지합니다.

`--hand-port`는 `--no-gripper`를 함의합니다. RH56이 툴 플랜지를 차지하고 있어서 UFACTORY 그리퍼
호출은 컨트롤러 에러 19를 래치시킵니다. 2지 그리퍼가 실제로 함께 장착된 경우에만 `--gripper`를 주세요.

| 플래그 | 기본값 | 비고 |
|---|---|---|
| `--scale`, `--depth-scale` | 3.0, 0.4 | D435 metric depth를 쓰면 `1.0`/`1.0` |
| `--tcp-speed` | 100 mm/s | 초기 실행은 80 이하 유지 |
| `--primary` | `right` | 검출기의 좌우 판정도 덮어씀 — 엄지 회전 방향을 결정 |
| `--min-cutoff`, `--beta` | 1.0, 0.02 | One-Euro 스무딩; cutoff가 낮을수록 부드럽고 지연 증가 |
| `--hand-rate` | 20 Hz | 손가락 명령 레이트 상한(RS485는 프레임 사이 여유가 필요) |
| `--hand-id`, `--hand-baud` | 1, 115200 | RH56 Modbus 주소 / 보드레이트 |
| `--pose-source` | `wilor` | `glove`로 두면 손 자세를 모캡 장갑에서 받음(2번 방법) |
| `--glove-port`, `--glove-host` | 7012, — | UDP 수신 포트; `--glove-host`를 주면 TCP로 접속 |
| `--glove-hand` | `--primary`를 따름 | 어느 쪽 장갑을 읽을지 |
| `--glove-timeout`, `--glove-wait` | 0.3 s, 5 s | 텔레옵 중 stale 판정 기준 / 시작 시 첫 프레임 대기 |
| `--glove-align-frames` | 30 | 재획득마다 장갑→카메라 회전을 평균할 프레임 수 |

그 밖의 튜닝 값은 코드에 있습니다. 워크스페이스 박스와 틱당 최대 이동량은
`src/control/safety.py`(`DEFAULT_WORKSPACE`, `max_step_m`), 홈 포즈는
`src/control/xarm_controller.py`(`HOME_Q`), 손가락 명령 범위와 지터 방지 데드밴드는
`src/control/inspire_hand.py`(`CMD_OPEN`/`CMD_CLOSED`, `min_delta`).

## 5. 안전

- **물리 E-stop을 손 닿는 곳에.** 카메라는 안전 장치가 아닙니다.
- **`src/control/safety.py`의 `DEFAULT_WORKSPACE`를** 사용 셀에서 도달 가능하고 충돌 없는 박스로
  설정하세요. 타깃은 여기에 하드 클램프됩니다.
- **천천히 시작** — `--tcp-speed 80` 이하, `max_step_m`도 작게(기본 0.02 m/tick).
- **`--hand-port`는 팔이 dry-run이어도 실제 손가락을 움직입니다.** 먼저 `hand-test`로 벤치 테스트를
  하세요.
- **추적을 놓치면 핸드는 마지막 위치를 유지합니다.** 쥔 물체를 떨어뜨리지는 않지만, 펴지도 않습니다.

## 6. 문제 해결

- **`no reply from the hand ... after 3 probes`** — 전원부터 확인하세요. 전원이 꺼진 핸드는 열린
  자세로 있어서, 카메라 쪽에서 보면 "완벽히 추적하는데 한 번도 쥐지 않는 텔레옵"과 구분되지 않습니다.
  그다음 RS485 A/B 극성, `--hand-id`, `--hand-baud`, 포트를 확인합니다.
- **`ControllerError, code: 19` / `set_servo_cartesian_aa -> code=1`** — 응답할 그리퍼가 없는
  툴 RS485 버스에 그리퍼 호출이 나간 경우입니다. `--hand-port`로 실행하세요(`--no-gripper` 함의).
  UFACTORY Studio에서 바꿀 것은 없습니다. `connect()`가 래치된 기존 에러를 지웁니다.
- **`Permission denied: /dev/ttyUSB0`** — `sudo usermod -aG dialout $USER` 후 재로그인.
- **시작할 때 핸드는 열리는데 손가락이 안 움직임** — 링크는 정상이고 타깃이 문제입니다. 실행 중
  추적 프레임 150개마다, 그리고 종료 시 `dex ratio range:`를 찍습니다. 범위가 한 값에 고정되어 있으면
  캘리브레이션이 포화한 것입니다. `--hand-calib none`으로 확인한 뒤 `hand-calib`을 다시 실행하세요.
- **손가락이 끝까지 닫히지 않음** — 종료 시 출력되는 DOF별 `dex raw vs calib` 표를 보세요(카메라
  오버레이에도 비율 막대 옆에 실시간 raw 각도가 표시됩니다). 주먹의 raw 각도가 캘리브레이션된 닫힘
  각도에 못 미치면 비율이 1.0에 도달하지 못해 핸드가 완전히 닫히지 못합니다. 텔레옵에서 쓰는 것과
  같은 세기의 주먹으로 `hand-calib`을 다시 실행하세요. WiLoR는 꽉 쥔 주먹의 curl을 과소추정하므로,
  닫힘 캡처는 해부학적 추정치가 아니라 WiLoR 자신의 추정값에서 나와야 합니다.
- **`no glove data on udp :7012 after 5s`** — Axis Studio가 이 PC에 닿지 않고 있습니다. BVH
  Broadcasting이 켜져 있고 *Destination*이 **이** PC의 주소와 포트 `7012`(루프백 아님, 노트북 자기
  주소도 아님)인지, 윈도우에서 `ping`이 되는지, 윈도우 방화벽이 송신을 막고 있지 않은지 확인하세요.
  `--glove-port`는 Destination 포트와 일치해야 하고, 텔레옵은 *Destination*에 적은 그 머신에서
  돌아야 합니다.
- **`ping`은 되는데 `tcpdump`에 UDP가 안 찍힘** — ICMP는 통과시키고 스트림은 버리는 구간이 있습니다.
  가능성 순서대로: 윈도우 방화벽(Axis Studio 아웃바운드를 개인 네트워크에서 허용), 이쪽 `ufw`
  (`sudo ufw allow from <윈도우 서브넷> to any port 7012 proto udp`), 그다음 두 서브넷 사이 공유기
  정책 — 마지막 경우는 랜선 직결(§2.1 a)로 아예 우회됩니다.
- **장갑은 연결되는데 본이 안 옴** — 반대쪽 손이 스트리밍되고 있거나, Axis Studio가 손/장갑 워킹
  모드가 아닙니다. `--glove-hand left|right`를 주세요. `glove-test`가 시작 시 스트림에서 빠진 본을
  나열합니다.
- **장갑 손가락은 되는데 팔 회전이 이상함** — 장갑→카메라 정합이 나쁜 프레임에서 잠겼습니다. 추적을
  놓았다가 다시 잡으면(재추정됩니다) 되고, `--glove-align-frames`를 키워도 됩니다. `--pos-only`는
  손 자세를 아예 무시합니다.
- **엉뚱한 손가락이 움직임** — DOF 순서는 `[little, ring, middle, index, thumb_bend, thumb_rot]`
  입니다. `hand-test`로 각 인덱스가 어느 물리 손가락을 구동하는지 확인하세요.
- **엄지만 반대 방향으로 회전** — 손바닥 법선의 부호는 검출기의 좌우 판정에서 옵니다.
  `--primary right` 또는 `--primary left`로 사용하는 손을 명시하세요.
- **손목 depth가 이상함** — depth 융합은 `--source realsense`에서만 동작합니다. 웹캠이나 영상에서는
  손목이 monocular라 스케일이 확정되지 않으며, 그래서 그쪽은 `--scale 3`을 씁니다.
- **추론이 느림(<20 fps)** — eager PyTorch에서는 정상입니다(3090에서 프레임당 ~50 ms). 제어 루프는
  퍼셉션과 분리되어 있습니다.
