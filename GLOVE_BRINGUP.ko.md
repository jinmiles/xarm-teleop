# 장갑(method 2) 브링업 현황 — 2026-09-16

이 장비에서 모캡 장갑 경로를 올리는 중에 남긴 스냅샷입니다. 일반적인 설치 절차는
[README.ko.md](README.ko.md) §2.1에 있고, 이 파일은 **이 rig의 실제 주소와 지금 막혀 있는 지점**만
담습니다. 브링업이 끝나면 지워도 됩니다.

> 이어서 작업하는 경우: 아래 "지금 할 일"부터 보면 됩니다. 코드는 이미 다 들어가 있고,
> 남은 것은 윈도우 쪽 송출을 살리는 일뿐입니다.

## 1. 장비와 주소

**텔레옵 PC** — `unist@unist-System-Product-Name`, repo `~/Test_hand/xarm-teleop`

| 인터페이스 | 주소 | 용도 |
|---|---|---|
| `eno1` | `10.20.26.115/24` | 랩 네트워크, 기본 경로 |
| `enx00e04c681f0b` | `192.168.2.15/24` | **노트북 직결 랜선**, 링크 UP |
| `enx00e04c2b5f28` | 없음 | NO-CARRIER, 미사용 |

**윈도우 노트북** — Wi-Fi `192.168.0.46`(인터넷), 이더넷 `192.168.2.16`(직결선, 게이트웨이·DNS 비움),
Axis Studio 실행.

`192.168.2.15`는 `ip addr add`로 넣은 런타임 설정이라 **재부팅하면 사라집니다.** 영구화:

```bash
sudo nmcli con add type ethernet ifname enx00e04c681f0b con-name glove-link \
    ipv4.method manual ipv4.addresses 192.168.2.15/24
sudo nmcli con up glove-link
```

게이트웨이는 주지 않습니다 — 기본 경로는 `eno1`에 그대로 둬야 랩 네트워크가 살아 있습니다.

## 2. Axis Studio 설정값

Settings → Working Mode = **Hand**, 그리고 Settings → BVH Broadcasting 토글 ON.

| 항목 | 값 |
|---|---|
| Frame Format → Type | `Binary` (*Use old header format* 해제) |
| Sync | `GenLock` 체크 |
| Skeleton | `Axis Studio` |
| BVH Format → Rotation | `XYZ` |
| BVH Format → Displacement | **체크** — 본 길이가 이 옵션으로 실려 옵니다 |
| Coordinate system | `OPT` |
| Protocol | `UDP` |
| Local Address | `192.168.2.16` : ~~`7001`~~ → **다른 포트로 변경 필요** (아래 3절) |
| Destination Address | `192.168.2.15` : `7012` ← 이 포트는 고정 |

## 3. 지금 막혀 있는 지점

**Axis Studio가 `192.168.2.16:7001` 바인드에 실패합니다 (access denied).** 송신 소켓이 아예 열리지
않아서 패킷이 노트북 밖으로 나가지 못했고, 그래서 우분투에서 `tcpdump`가 계속 `0 packets`였습니다.
방화벽은 바인드를 막지 않으므로 이건 포트 점유/예약 문제입니다.

조치 순서:

1. **Local Port를 `17001` 등으로 변경.** 이 값은 송신 소켓 번호라 무엇이든 상관없습니다.
   Destination 포트 `7012`만 그대로 두면 수신부는 영향받지 않습니다.
2. 그래도 실패하면 Axis Studio를 **관리자 권한으로 실행**한 뒤 설정을 다시 넣고 OK.
3. 원인 확인 (윈도우 cmd):
   ```cmd
   netsh int ipv4 show excludedportrange protocol=udp
   netstat -ano | findstr :7001
   ```
   Hyper-V / WSL2 / Docker가 포트 블록을 예약해두면 그 범위의 포트는 무조건 access denied입니다.

## 4. 미해결 관측

- `ip neigh`에서 `192.168.2.16`이 `INCOMPLETE`(ARP 무응답)로 나온 적이 있습니다. 바인드 실패와 별개
  문제일 수 있으니, 윈도우에서 `ipconfig`로 **케이블이 꽂힌 그 어댑터에** `192.168.2.16`이 실제로
  붙어 있는지 다시 확인하세요. USB 랜카드를 쓴다면 내장 이더넷이 아니라 그 어댑터에 설정해야 합니다.
- **`ping`으로 판정하지 마세요.** 윈도우 방화벽이 들어오는 ICMP를 기본 차단하므로, 주소가 다 맞아도
  ping은 실패합니다. 판정은 `tcpdump`로만 합니다.

## 5. 검증 순서

```bash
cd ~/Test_hand/xarm-teleop && git pull       # MocapApi 라이브러리 포함, 외부 repo 불필요
sudo tcpdump -i any -n udp port 7012         # 어느 NIC으로 들어와도 정상 (eno1이어도 됨)
python scripts/teleop.py glove-test          # 손가락을 하나씩 굽혀 해당 DOF만 움직이는지 확인
```

`tcpdump`에 찍히는데 `glove-test`가 조용하면 워킹 모드(Hand)나 손 선택(`--glove-hand`) 문제입니다.
둘 다 조용하면 3절로 돌아갑니다.

## 6. 코드 상태

`master` = `ae7a2a3`. 2번 방법(`--pose-source glove`)은 전체 구현·검증이 끝났고, Noitom MocapApi
라이브러리는 `vendor/noitom/`에 포함되어 있어 이 repo를 clone하는 것만으로 동작합니다
(`mocap_ros_py` 같은 외부 체크아웃 불필요).

glove-test가 통과하면 다음 순서로 진행합니다:

```bash
python scripts/teleop.py hand-calib --pose-source glove          # 카메라 없이 가능
python scripts/teleop.py sim --source realsense --pose-source glove \
    --hand-port /dev/ttyUSB0 --display
python scripts/teleop.py teleop --execute --ip <XARM_IP> --source realsense \
    --scale 1.0 --depth-scale 1.0 --tcp-speed 80 \
    --pose-source glove --hand-port /dev/ttyUSB0 --display
```

이 rig에서 아직 확인되지 않은 것: 실제 장갑 데이터로의 동작 전체(스트림이 아직 안 들어옴), D435·xArm7·
RH56 연결 상태.
